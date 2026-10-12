"""Demo do pipeline de audio.

Uso:
    python scripts/demo_audio.py <caminho_do_audio>

Sem argumento, gera audios sinteticos via `gen_synthetic_audio.py` e
processa o cenario `normal`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gen_synthetic_audio import OUTPUT_DIR  # noqa: E402
from scripts.gen_synthetic_audio import main as gerar

from src.audio.pipeline import AudioPipeline  # noqa: E402
from src.audio.types import AudioAnalysis  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo_audio")


def imprimir_resumo(analise: AudioAnalysis) -> None:
    """Imprime resumo conciso do `AudioAnalysis`."""
    print(f"Tipo: {type(analise).__name__}")
    print(f"Transcricao: {analise.transcription[:80] or '(vazia)'}")
    print(f"Segmentos: {len(analise.segments)}")
    if analise.acoustic_features:
        feats = analise.acoustic_features
        print(
            f"Features: dur={feats.duration_s:.1f}s, pitch={feats.pitch_mean_hz:.0f}Hz "
            f"(+/- {feats.pitch_std_hz:.0f}), rms={feats.energy_rms:.3f}, "
            f"jitter={feats.jitter:.3f}, shimmer={feats.shimmer:.3f}"
        )
    if analise.emotion:
        print(f"Emocao: {analise.emotion.label} ({analise.emotion.confidence:.2f})")
    else:
        print("Emocao: N/A (modelo indisponivel)")
    if analise.sentiment:
        print(f"Sentimento: {analise.sentiment.label} ({analise.sentiment.confidence:.2f})")
    else:
        print("Sentimento: N/A (Azure Language nao configurado)")
    print(f"Key phrases: {analise.key_phrases or '(nenhuma)'}")


def main() -> None:
    if len(sys.argv) >= 2:
        audio_path = Path(sys.argv[1])
    else:
        audio_path = OUTPUT_DIR / "audio_normal.wav"
        if not audio_path.exists():
            gerar()

    pipeline = AudioPipeline()
    analise = pipeline.process(audio_path)
    imprimir_resumo(analise)
    if not isinstance(analise, AudioAnalysis):
        logger.error("Saida nao e AudioAnalysis")
        sys.exit(1)


if __name__ == "__main__":
    main()
