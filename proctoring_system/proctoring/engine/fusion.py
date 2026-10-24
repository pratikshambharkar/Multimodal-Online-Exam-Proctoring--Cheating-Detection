"""Decision-level fusion engine (manuscript Sec. 3.7).

Fuses the per-channel evidence produced by the vision/audio engines into a
single per-frame label via a trained SVM (scikit-learn ``SVC`` with
``probability=True``) and maintains the exponentially weighted moving-average
(EWMA) risk score R_t. When the trained artifacts are absent the engine
degrades to a deterministic rule-based combiner with identical output
semantics, so the pipeline keeps working on a fresh checkout.
"""

from django.conf import settings


def _clamp01(value):
    """Clamp a numeric value into the closed interval [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _num(ch, key, default=0.0):
    """Read a numeric channel value, treating a missing key or None as absent."""
    value = ch.get(key)
    return float(default) if value is None else float(value)


class FusionEngine:
    """SVM decision fusion plus EWMA risk accumulation.

    The SVM was trained on 10-dimensional feature vectors (see
    :meth:`features_from_channels` for the exact order) with the string
    labels ``'cheating'`` and ``'normal'``; the index of the ``'cheating'``
    class is resolved dynamically from ``svm.classes_`` so the class order
    used at training time never matters here.
    """

    #: P(cheat) at or above this maps to the 'violation' label.
    VIOLATION_PROB = 0.75
    #: P(cheat) at or above this (but below VIOLATION_PROB) maps to 'suspicious'.
    SUSPICIOUS_PROB = 0.45

    def __init__(self):
        self.svm = None
        self.scaler = None
        self._cheat_idx = None
        self._load_models()

    def _load_models(self):
        """Load the joblib artifacts if present; stay in fallback mode otherwise."""
        try:
            import joblib  # imported here so `manage.py migrate` never needs it
        except Exception:
            return
        svm_path = settings.ML_MODELS_DIR / "svm_fusion.joblib"
        scaler_path = settings.ML_MODELS_DIR / "svm_scaler.joblib"
        if not (svm_path.exists() and scaler_path.exists()):
            return
        try:
            svm = joblib.load(svm_path)
            scaler = joblib.load(scaler_path)
            cheat_idx = list(svm.classes_).index("cheating")
        except Exception:
            return
        self.svm = svm
        self.scaler = scaler
        self._cheat_idx = cheat_idx

    def features_from_channels(self, ch):
        """Build the 10-dim feature vector in the exact order used at training.

        Order: [face_count, looking_away(0/1), abs(yaw), abs(pitch), ear,
        phone_conf, book_conf, laptop_conf, extra_persons, cnn_cheat_prob].
        Missing values default to a neutral single attentive face.
        """
        return [
            _num(ch, "face_count", 1.0),
            1.0 if ch.get("looking_away") else 0.0,
            abs(_num(ch, "yaw")),
            abs(_num(ch, "pitch")),
            _num(ch, "ear", 0.28),
            _num(ch, "phone_conf"),
            _num(ch, "book_conf"),
            _num(ch, "laptop_conf"),
            _num(ch, "extra_persons"),
            _num(ch, "cnn_cheat_prob"),
        ]

    def classify(self, ch):
        """Classify one frame's channel dict.

        Returns ``(label, p_cheat)`` where label is 'normal', 'suspicious' or
        'violation'. Uses the trained SVM when available and falls back to the
        rule-based combiner otherwise (or if inference fails at runtime).
        """
        p_cheat = None
        if self.svm is not None and self.scaler is not None:
            try:
                scaled = self.scaler.transform([self.features_from_channels(ch)])
                p_cheat = float(self.svm.predict_proba(scaled)[0][self._cheat_idx])
            except Exception:
                p_cheat = None
        if p_cheat is None:
            p_cheat = self._rule_based_prob(ch)
        p_cheat = _clamp01(p_cheat)
        if p_cheat >= self.VIOLATION_PROB:
            label = "violation"
        elif p_cheat >= self.SUSPICIOUS_PROB:
            label = "suspicious"
        else:
            label = "normal"
        return label, p_cheat

    @staticmethod
    def _rule_based_prob(ch):
        """Fallback P(cheat): max-combined weighted evidence, clamped to [0, 1]."""
        face_absent = 1.0 if _num(ch, "face_count", 1.0) == 0 else 0.0
        looking_away = 1.0 if ch.get("looking_away") else 0.0
        contributions = (
            _num(ch, "phone_conf") * 0.9,
            _num(ch, "extra_persons") * 0.8,
            looking_away * 0.35,
            _num(ch, "cnn_cheat_prob") * 0.5,
            face_absent * 0.4,
        )
        return _clamp01(max(contributions))

    def channel_scores(self, ch, browser_score=0.0):
        """Compute the per-channel evidence scores s_i in [0, 1].

        Keys match ``settings.PROCTORING['CHANNEL_WEIGHTS']``: identity,
        gaze, objects, cnn, audio, browser.
        """
        identity = 1.0 if ch.get("identity_verified") is False else 0.0
        face_count = int(_num(ch, "face_count", 1.0))
        if face_count == 0:
            identity = max(identity, 0.6)
        elif face_count > 1:
            identity = max(identity, 0.8)
        objects = max(
            _num(ch, "phone_conf"),
            0.7 * _num(ch, "book_conf"),
            0.9 if _num(ch, "extra_persons") > 0 else 0.0,
        )
        return {
            "identity": _clamp01(identity),
            "gaze": _clamp01(_num(ch, "gaze_off_fraction")),
            "objects": _clamp01(objects),
            "cnn": _clamp01(_num(ch, "cnn_cheat_prob")),
            "audio": _clamp01(_num(ch, "audio_speech_prob")),
            "browser": _clamp01(browser_score),
        }

    def update_risk(self, prev_risk, scores):
        """EWMA risk update: R_t = lambda * R_{t-1} + (1 - lambda) * sum(w_i * s_i)."""
        cfg = settings.PROCTORING
        lam = float(cfg["RISK_LAMBDA"])
        weights = cfg["CHANNEL_WEIGHTS"]
        instant = sum(
            float(weight) * _clamp01(scores.get(name, 0.0))
            for name, weight in weights.items()
        )
        previous = float(prev_risk) if prev_risk else 0.0
        return _clamp01(lam * previous + (1.0 - lam) * instant)
