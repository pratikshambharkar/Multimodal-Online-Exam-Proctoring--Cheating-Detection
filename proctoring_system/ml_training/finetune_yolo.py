"""
RAIM - finetune_yolo.py (optional step)

Fine-tunes YOLOv8n on the Roboflow 'cheating online exam' detection set
(2 classes: cheat / normal) and copies the best weights to
ml_models/yolo_cheat.pt.

The set's data.yaml uses relative '../' paths that break when training from
another working directory, so a corrected yaml with absolute train/val/test
paths is written to the output directory first.

Usage (from project root):
    venv\\Scripts\\python.exe ml_training\\finetune_yolo.py [--epochs 10]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "DataSet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "ml_models"
YOLO_SET = "cheating online exam.v3i.yolov8"
RUN_NAME = "yolo_finetune"


def parse_args():
    """Command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on the cheat/normal detection set.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory containing the dataset folders "
                             f"(default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Artifact output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--device", default="cpu",
                        help="Ultralytics device string (default: cpu)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8 - CPU-friendly)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Dataloader workers (default: 2)")
    return parser.parse_args()


def write_absolute_yaml(yolo_dir: Path, out_dir: Path) -> Path:
    """Rewrite data.yaml with absolute split paths; return the new yaml path."""
    import yaml

    source = yolo_dir / "data.yaml"
    if not source.exists():
        raise FileNotFoundError(f"data.yaml not found: {source}")
    original = yaml.safe_load(source.read_text(encoding="utf-8"))

    corrected = {
        "train": (yolo_dir / "train" / "images").resolve().as_posix(),
        "val": (yolo_dir / "valid" / "images").resolve().as_posix(),
        "test": (yolo_dir / "test" / "images").resolve().as_posix(),
        "nc": original["nc"],
        "names": original["names"],
    }
    for key in ("train", "val", "test"):
        if not Path(corrected[key]).is_dir():
            raise FileNotFoundError(f"Missing image directory: {corrected[key]}")

    target = out_dir / "yolo_data_abs.yaml"
    target.write_text(yaml.safe_dump(corrected, sort_keys=False),
                      encoding="utf-8")
    print(f"Wrote corrected dataset yaml: {target}")
    print(f"  classes ({corrected['nc']}): {corrected['names']}")
    return target


def main():
    args = parse_args()
    started = time.perf_counter()

    yolo_dir = (args.data_root / YOLO_SET).resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {yolo_dir}")
    print(f"Output  : {out_dir}")
    yaml_path = write_absolute_yaml(yolo_dir, out_dir)

    from ultralytics import YOLO

    # Reuse the COCO nano weights already shipped with the app if present.
    local_weights = out_dir / "yolov8n.pt"
    base_weights = str(local_weights) if local_weights.exists() else "yolov8n.pt"
    print(f"Base weights: {base_weights}")

    model = YOLO(base_weights)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        project=str(out_dir),
        name=RUN_NAME,
        exist_ok=True,
    )

    best = out_dir / RUN_NAME / "weights" / "best.pt"
    if not best.exists() and getattr(model, "trainer", None) is not None:
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print(f"ERROR: best.pt not found under {out_dir / RUN_NAME} - "
              "training did not complete.")
        return 1

    target = out_dir / "yolo_cheat.pt"
    shutil.copy2(best, target)
    print(f"\nCopied best weights: {best} -> {target}")

    print("Validating fine-tuned weights on the valid split...")
    best_model = YOLO(str(best))
    metrics = best_model.val(
        data=str(yaml_path),
        imgsz=args.imgsz,
        device=args.device,
        project=str(out_dir),
        name=f"{RUN_NAME}_val",
        exist_ok=True,
    )
    print(f"\nValidation mAP50    : {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95 : {metrics.box.map:.4f}")
    print(f"Elapsed: {(time.perf_counter() - started) / 60:.1f} min")
    print("\nTo make the app use these weights, set "
          "PROCTORING['YOLO_WEIGHTS'] = 'yolo_cheat.pt' (trunk change).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user - partial run directory left in place.")
        sys.exit(130)
