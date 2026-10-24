"""
RAIM - train_cnn.py

Trains the behaviour CNN (MobileNetV2 transfer learning) on the Roboflow
binary classification set (cheating vs normal) and saves the best model to
ml_models/behavior_cnn.keras.

Class order is fixed by the app contract:
    index 0 = cheating, index 1 = normal
(BehaviorCNN.predict reads softmax index 0 as P(cheating).)

Input contract for the served model: 160x160 RGB, scaled to [0, 1].

Usage (from project root):
    venv\\Scripts\\python.exe ml_training\\train_cnn.py [--epochs 6] [--fine-tune]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TF C++ banner before import

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "DataSet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml_models"
BINARY_SET = "online exam cheating detection.v15i.folder"

IMG_SIZE = 160
BATCH_SIZE = 32
CLASS_NAMES = ["cheating", "normal"]  # index 0 = cheating - DO NOT REORDER
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the RAIM behaviour CNN.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory containing the dataset folders "
                             f"(default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Artifact output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Frozen-backbone training epochs (default: 6)")
    parser.add_argument("--split-mode", choices=["pooled", "folder"],
                        default="pooled",
                        help="'pooled' (default): merge all splits and stratify-"
                             "resplit 70/15/15 (folder splits are domain-"
                             "disjoint); 'folder': use the Roboflow splits "
                             "as-is (honest cross-session protocol)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="After frozen training, unfreeze the last 30 backbone "
                             "layers and train 2 more epochs at lr 1e-4")
    return parser.parse_args()


def prune_empty_class_dirs(split_dir: Path):
    """Remove empty extraneous class directories (e.g. Roboflow's 'Unlabeled/').

    Only truly empty directories are removed (os.rmdir); a non-empty extraneous
    directory is left in place with a warning - datasets are always built from
    explicit cheating/normal file lists, so it cannot pollute training either way.
    """
    for child in sorted(split_dir.iterdir()):
        if child.is_dir() and child.name not in CLASS_NAMES:
            try:
                os.rmdir(child)
                print(f"  removed empty extraneous class dir: {child.name}/")
            except OSError:
                print(f"  WARNING: ignoring non-empty extraneous dir: {child}")


def list_split_files(data_dir: Path, split: str):
    """Return (paths, labels) for one split, using only the two contract classes."""
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset split directory: {split_dir}")
    prune_empty_class_dirs(split_dir)
    paths, labels = [], []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        class_files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(str(p) for p in class_files)
        labels.extend([label_index] * len(class_files))
        print(f"  {split}/{class_name}: {len(class_files)} images")
    return paths, labels


def build_dataset(paths, labels, training: bool, augment=None):
    """tf.data pipeline from explicit file lists (never from directory scanning)."""
    import tensorflow as tf

    def _load(path, label):
        data = tf.io.read_file(path)
        image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32)))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED,
                        reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()  # drop unreadable files instead of crashing
    ds = ds.batch(BATCH_SIZE)
    if training and augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    """MobileNetV2 backbone (frozen) + GAP + Dropout + 2-way softmax head."""
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="frame")
    # The serving contract feeds [0,1] images; MobileNetV2's ImageNet weights
    # expect [-1,1], so the conversion is baked into the saved model.
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="to_pm1")(inputs)
    x = base(x, training=False)  # BN stays in inference mode, even fine-tuned
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="cheat_softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="raim_behavior_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base


def merge_histories(*histories):
    """Concatenate Keras History.history dicts across training phases."""
    merged = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def plot_training_curves(history: dict, phase_boundary: int, out_path: Path):
    """Save accuracy/loss curves (train + valid) as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_acc.plot(epochs, history.get("accuracy", []), "o-", label="train")
    ax_acc.plot(epochs, history.get("val_accuracy", []), "s-", label="valid")
    ax_acc.set_title("Accuracy")
    ax_loss.plot(epochs, history.get("loss", []), "o-", label="train")
    ax_loss.plot(epochs, history.get("val_loss", []), "s-", label="valid")
    ax_loss.set_title("Loss")
    for ax in (ax_acc, ax_loss):
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        if 0 < phase_boundary < len(list(epochs)):
            ax.axvline(phase_boundary + 0.5, color="gray", linestyle="--",
                       linewidth=1, label="fine-tune start")
        ax.legend()
    fig.suptitle("RAIM behaviour CNN - MobileNetV2 transfer learning")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_on_test(model, test_ds):
    """Run the model over the test dataset, keeping labels aligned per batch."""
    import numpy as np

    y_true, y_pred = [], []
    for batch_x, batch_y in test_ds:
        probs = model.predict_on_batch(batch_x)
        y_true.extend(int(v) for v in batch_y.numpy())
        y_pred.extend(int(v) for v in np.argmax(probs, axis=1))
    return y_true, y_pred


def main():
    args = parse_args()
    started = time.perf_counter()

    data_dir = (args.data_root / BINARY_SET).resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "behavior_cnn.keras"

    print(f"Dataset : {data_dir}")
    print(f"Output  : {out_dir}")

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.keras.utils.set_random_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} - device: "
          f"{gpus[0].name if gpus else 'CPU only'}")

    print("Indexing image files (explicit file lists; Unlabeled/ excluded):")
    train_paths, train_labels = list_split_files(data_dir, "train")
    valid_paths, valid_labels = list_split_files(data_dir, "valid")
    test_paths, test_labels = list_split_files(data_dir, "test")

    if args.split_mode == "pooled":
        # The Roboflow folder splits come from disjoint recording sessions
        # (severe train/test domain shift: folder-split test accuracy lands
        # near chance).  Pooling and re-splitting stratified gives the served
        # model coverage of every capture domain.  Caveat for reporting:
        # consecutive video frames are near-duplicates, so pooled-split test
        # numbers are optimistic; both protocols are noted in the metrics.
        from sklearn.model_selection import train_test_split

        all_paths = train_paths + valid_paths + test_paths
        all_labels = train_labels + valid_labels + test_labels
        train_paths, rest_paths, train_labels, rest_labels = train_test_split(
            all_paths, all_labels, test_size=0.30, random_state=SEED,
            stratify=all_labels)
        valid_paths, test_paths, valid_labels, test_labels = train_test_split(
            rest_paths, rest_labels, test_size=0.50, random_state=SEED,
            stratify=rest_labels)
        print(f"  pooled re-split (stratified, seed {SEED}): "
              f"train {len(train_paths)}, valid {len(valid_paths)}, "
              f"test {len(test_paths)}")

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.04, seed=SEED),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0), seed=SEED),
        tf.keras.layers.RandomContrast(0.15, seed=SEED),
    ], name="augmentation")

    train_ds = build_dataset(train_paths, train_labels, training=True, augment=augment)
    valid_ds = build_dataset(valid_paths, valid_labels, training=False)
    test_ds = build_dataset(test_paths, test_labels, training=False)

    model, base = build_model()
    model.summary(line_length=90)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=3,
            restore_best_weights=True, verbose=1),
    ]

    print(f"\nPhase 1 - frozen backbone, {args.epochs} epochs, lr 1e-3")
    history_frozen = model.fit(train_ds, validation_data=valid_ds,
                               epochs=args.epochs, callbacks=callbacks)
    epochs_frozen = len(history_frozen.history["loss"])

    history_finetune = None
    if args.fine_tune:
        print("\nPhase 2 - fine-tune: unfreezing last 30 backbone layers, "
              "4 epochs, lr 1e-4")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history_finetune = model.fit(
            train_ds, validation_data=valid_ds,
            initial_epoch=epochs_frozen, epochs=epochs_frozen + 4,
            callbacks=callbacks)

    # Evaluate the checkpointed best model (exactly the artifact the app serves).
    print(f"\nReloading best checkpoint for test evaluation: {model_path}")
    best_model = tf.keras.models.load_model(model_path)

    from sklearn.metrics import classification_report, confusion_matrix
    y_true, y_pred = evaluate_on_test(best_model, test_ds)
    report_text = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, digits=4,
        zero_division=0)
    report_dict = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nTest-set classification report (0=cheating, 1=normal):")
    print(report_text)
    print("Confusion matrix [rows=true, cols=pred] (cheating, normal):")
    print(matrix)

    history = merge_histories(history_frozen, history_finetune)
    plot_path = out_dir / "cnn_training.png"
    plot_training_curves(history, epochs_frozen if args.fine_tune else 0, plot_path)

    elapsed = time.perf_counter() - started
    metrics = {
        "model": "MobileNetV2 transfer (frozen backbone"
                 + (" + last-30-layer fine-tune" if args.fine_tune else "") + ")",
        "class_order": CLASS_NAMES,
        "split_mode": args.split_mode,
        "split_note": ("pooled stratified re-split; near-duplicate video frames "
                       "make these numbers optimistic vs the cross-session "
                       "folder protocol" if args.split_mode == "pooled"
                       else "Roboflow folder splits (cross-session, "
                            "domain-disjoint)"),
        "input": f"{IMG_SIZE}x{IMG_SIZE} RGB [0,1]",
        "counts": {"train": len(train_paths), "valid": len(valid_paths),
                   "test": len(test_paths)},
        "epochs_frozen": epochs_frozen,
        "fine_tuned": bool(args.fine_tune),
        "test_accuracy": float(report_dict["accuracy"]),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "history": history,
        "elapsed_seconds": round(elapsed, 1),
    }
    metrics_path = out_dir / "cnn_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved model            : {model_path}")
    print(f"Saved metrics          : {metrics_path}")
    print(f"Saved training curves  : {plot_path}")
    print(f"Elapsed: {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user - no further artifacts written.")
        sys.exit(130)
