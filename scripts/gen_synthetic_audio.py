"""Gera audios sinteticos para teste do pipeline de audio.

Sem TTS Azure, gera 4 `.wav` curtos com caracteristicas acusticas
distintas para validar `extract_features` e o classificador de emocao.
Cada audio e nao-falado (ondas + ruido); a transcricao via Whisper
retornara texto vazio ou nonsense, o que e aceitavel para smoke.

Uso:
    python scripts/gen_synthetic_audio.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gen_synthetic_audio")

OUTPUT_DIR = Path("data/synthetic")
SAMPLE_RATE = 16_000
DURATION_S = 3.0

# Cada cenario define caracteristicas que tornam suas features acusticas distintas.
SCENARIOS: dict[str, dict[str, float]] = {
    "normal": {"pitch_hz": 220.0, "noise": 0.05, "energy": 0.3, "vibrato_hz": 5.0},
    "ansiedade": {"pitch_hz": 380.0, "noise": 0.15, "energy": 0.5, "vibrato_hz": 8.0},
    "depressao": {"pitch_hz": 130.0, "noise": 0.03, "energy": 0.1, "vibrato_hz": 2.0},
    "monocordico": {"pitch_hz": 220.0, "noise": 0.02, "energy": 0.25, "vibrato_hz": 0.0},
}


def gerar_wav(cenario: str, params: dict[str, float], destino: Path) -> Path:
    """Gera um wav sintetico baseado nos parametros do cenario."""
    t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S), endpoint=False)
    if params["vibrato_hz"] > 0:
        modulacao = 1.0 + 0.05 * np.sin(2 * np.pi * params["vibrato_hz"] * t)
    else:
        modulacao = np.ones_like(t)
    onda = np.sin(2 * np.pi * params["pitch_hz"] * modulacao * t)
    ruido = np.random.default_rng(seed=hash(cenario) % 2**32).normal(0, params["noise"], len(t))
    sinal = params["energy"] * (onda + ruido)
    sinal = np.clip(sinal, -1.0, 1.0).astype(np.float32)
    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destino), sinal, SAMPLE_RATE, subtype="PCM_16")
    logger.info("Gerado %s (%.1fs, %d Hz, %s)", destino, DURATION_S, SAMPLE_RATE, cenario)
    return destino


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cenario, params in SCENARIOS.items():
        destino = OUTPUT_DIR / f"audio_{cenario}.wav"
        gerar_wav(cenario, params, destino)
    print(f"Gerados {len(SCENARIOS)} arquivos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
