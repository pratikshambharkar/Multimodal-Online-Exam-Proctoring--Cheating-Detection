"""Microphone-chunk audio analysis (manuscript Sec. 3.5).

Loads a recorded chunk (webm/ogg/wav), extracts energy and spectral features
and derives a heuristic speech probability plus a categorical label:
``silence | noise | speech | conversation``.

Decoding uses librosa (with a soundfile fallback), imported lazily.  All the
spectral features themselves are computed with numpy so the analyzer works
with either decoder.  Unreadable or corrupt files yield ``available=False``.
"""

import numpy as np
from django.conf import settings

_TARGET_SR = 16000
_FRAME_LEN = 512          # ~32 ms at 16 kHz
_HOP = 256
_EPS = 1e-10

# Heuristic constants (tuned for normalized float audio in [-1, 1]).
_RMS_SILENCE = 0.008      # below this the chunk is labelled "silence"
_RMS_GATE_LO = 0.004      # speech-prob gate ramps up from here ...
_RMS_GATE_HI = 0.020      # ... to full confidence here
_SPEECH_BAND = (300.0, 3000.0)   # telephone-quality speech band (Hz)
_BAND_RATIO_LO = 0.25     # band-energy ratio mapped 0..1 over this range
_BAND_RATIO_HI = 0.75
_ZCR_SPEECH_CENTER = 0.08  # typical per-sample ZCR for voiced speech @16k
_ZCR_SPEECH_SPAN = 0.10
_FLATNESS_NOISE = 0.5      # spectral flatness near 1 => noise-like
_CENTROID_SPEECH_HZ = 1600.0
_CENTROID_SPAN_HZ = 2400.0
# Weighted combination of the four spectral scores (sums to 1).
_W_BAND, _W_ZCR, _W_FLAT, _W_CENTROID = 0.45, 0.25, 0.15, 0.15
# Coefficient of variation of frame RMS above which sustained speech is
# treated as turn-taking ("conversation").
_TURN_TAKING_CV = 1.0

_NEUTRAL = {"available": False, "rms": 0.0, "speech_prob": 0.0, "label": "silence"}


def _clamp01(value):
    return min(1.0, max(0.0, float(value)))


class AudioAnalyzer:
    """Energy + spectral-feature analysis of one recorded audio chunk."""

    def __init__(self):
        self._speech_alert = float(settings.PROCTORING["AUDIO_SPEECH_PROB_ALERT"])

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_audio(path):
        """Decode to mono float; returns (samples, sample_rate) or (None, 0)."""
        try:
            import librosa

            y, sr = librosa.load(path, sr=_TARGET_SR, mono=True)
            return np.asarray(y, dtype=np.float64), int(sr)
        except Exception:
            pass
        try:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return data.mean(axis=1).astype(np.float64), int(sr)
        except Exception:
            return None, 0

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _frame(y):
        """Slice the signal into overlapping frames (n_frames, _FRAME_LEN)."""
        if y.size < _FRAME_LEN:
            y = np.pad(y, (0, _FRAME_LEN - y.size))
        n_frames = 1 + (y.size - _FRAME_LEN) // _HOP
        idx = np.arange(_FRAME_LEN)[None, :] + _HOP * np.arange(n_frames)[:, None]
        return y[idx]

    @staticmethod
    def _spectral_features(frames, sr):
        """Energy-weighted band ratio, centroid (Hz) and flatness."""
        window = np.hanning(_FRAME_LEN)
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
        power = spectrum ** 2
        freqs = np.fft.rfftfreq(_FRAME_LEN, d=1.0 / sr)

        total_power = float(power.sum()) + _EPS
        band_mask = (freqs >= _SPEECH_BAND[0]) & (freqs <= _SPEECH_BAND[1])
        band_ratio = float(power[:, band_mask].sum()) / total_power

        summed = power.sum(axis=0)
        centroid = float((freqs * summed).sum() / (summed.sum() + _EPS))

        # Per-frame flatness, weighted by frame energy so silence between
        # words does not dominate the statistic.
        geo_mean = np.exp(np.mean(np.log(power + _EPS), axis=1))
        arith_mean = power.mean(axis=1) + _EPS
        flat_frames = geo_mean / arith_mean
        weights = power.sum(axis=1)
        flatness = float((flat_frames * weights).sum() / (weights.sum() + _EPS))
        return band_ratio, centroid, flatness

    def _speech_probability(self, rms, zcr, band_ratio, centroid, flatness):
        """Weighted, clamped heuristic P(speech) in [0, 1]."""
        band_score = _clamp01(
            (band_ratio - _BAND_RATIO_LO) / (_BAND_RATIO_HI - _BAND_RATIO_LO)
        )
        zcr_score = _clamp01(
            1.0 - abs(zcr - _ZCR_SPEECH_CENTER) / _ZCR_SPEECH_SPAN
        )
        flat_score = _clamp01(1.0 - flatness / _FLATNESS_NOISE)
        centroid_score = _clamp01(
            1.0 - abs(centroid - _CENTROID_SPEECH_HZ) / _CENTROID_SPAN_HZ
        )
        base = (
            _W_BAND * band_score
            + _W_ZCR * zcr_score
            + _W_FLAT * flat_score
            + _W_CENTROID * centroid_score
        )
        # Attenuate toward zero when the chunk is barely above the noise floor.
        gate = _clamp01((rms - _RMS_GATE_LO) / (_RMS_GATE_HI - _RMS_GATE_LO))
        return _clamp01(base * gate)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze_file(self, path):
        """Analyze one audio chunk file; never raises.

        Returns ``{"available", "rms", "speech_prob", "label"}``.
        """
        try:
            y, sr = self._load_audio(path)
            if y is None or sr <= 0:
                return dict(_NEUTRAL)
            y = y[np.isfinite(y)]
            if y.size == 0:
                return dict(_NEUTRAL)

            rms = float(np.sqrt(np.mean(np.square(y))))
            zcr = float(np.mean(np.abs(np.diff(np.signbit(y).astype(np.int8)))))

            frames = self._frame(y)
            band_ratio, centroid, flatness = self._spectral_features(frames, sr)
            speech_prob = self._speech_probability(
                rms, zcr, band_ratio, centroid, flatness
            )

            if rms < _RMS_SILENCE:
                label = "silence"
            elif speech_prob >= self._speech_alert:
                # Bursty frame energy (speech interleaved with pauses) marks
                # turn-taking between two speakers.
                frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
                rms_cv = float(np.std(frame_rms) / (np.mean(frame_rms) + _EPS))
                label = "conversation" if rms_cv > _TURN_TAKING_CV else "speech"
            else:
                label = "noise"

            return {
                "available": True,
                "rms": round(rms, 4),
                "speech_prob": round(speech_prob, 3),
                "label": label,
            }
        except Exception:
            return dict(_NEUTRAL)
