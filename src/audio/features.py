"""Extracao de features acusticas via librosa.

Compute de pitch (F0), energia RMS, zero-crossing rate, jitter, shimmer e
13 coeficientes MFCC (media temporal). Saida em `AcousticFeatures`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.audio.types import AcousticFeatures

logger = logging.getLogger(__name__)


def extract_features(audio_path: Path, sample_rate: int = 16_000) -> AcousticFeatures:
    """Extrai features acusticas de um arquivo de audio.

    Args:
        audio_path: caminho do arquivo de audio.
        sample_rate: taxa de amostragem alvo (downsampling se necessario).
            16 kHz e o padrao para fala.

    Returns:
        `AcousticFeatures` com pitch, energia, jitter, shimmer e MFCC.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    duration_s = float(len(y) / sr)
    logger.info("Extraindo features de %s (%.2fs @ %d Hz)", audio_path, duration_s, sr)

    pitch_mean, pitch_std = _pitch_stats(y, sr)
    energy_rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    jitter = _jitter(y, sr)
    shimmer = _shimmer(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = [float(x) for x in np.mean(mfcc, axis=1)]

    return AcousticFeatures(
        duration_s=duration_s,
        pitch_mean_hz=pitch_mean,
        pitch_std_hz=pitch_std,
        energy_rms=energy_rms,
        zero_crossing_rate=zcr,
        jitter=jitter,
        shimmer=shimmer,
        mfcc_means=mfcc_means,
    )


def _pitch_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Calcula media e desvio do pitch usando librosa.pyin."""
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    voiced = f0[voiced_flag] if voiced_flag.any() else f0[~np.isnan(f0)]
    if len(voiced) == 0:
        return 0.0, 0.0
    return float(np.nanmean(voiced)), float(np.nanstd(voiced))


def _jitter(y: np.ndarray, sr: int) -> float:
    """Aproxima jitter como variacao relativa de periodos consecutivos."""
    import librosa

    f0, _, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 2:
        return 0.0
    periods = 1.0 / f0
    diffs = np.abs(np.diff(periods))
    mean_period = float(np.mean(periods))
    if mean_period <= 0:
        return 0.0
    return float(np.mean(diffs) / mean_period)


def _shimmer(y: np.ndarray, sr: int) -> float:
    """Aproxima shimmer como variacao relativa de amplitude entre quadros."""
    import librosa

    rms = librosa.feature.rms(y=y)[0]
    if len(rms) < 2:
        return 0.0
    diffs = np.abs(np.diff(rms))
    mean_rms = float(np.mean(rms))
    if mean_rms <= 0:
        return 0.0
    return float(np.mean(diffs) / mean_rms)
