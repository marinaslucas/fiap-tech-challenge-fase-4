"""Demo do pipeline de video.

Uso:
    python scripts/demo_video.py <caminho_do_video>

Se nenhum caminho for passado, gera um video sintetico curto (5 segundos
de retangulo se movendo) em `data/synthetic/demo_video.mp4` e processa-o.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Permite rodar `python scripts/demo_video.py` direto da raiz do projeto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video.pipeline import VideoPipeline  # noqa: E402
from src.video.types import VideoEvent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo_video")

DEFAULT_DEMO_PATH = Path("data/synthetic/demo_video.mp4")


def gerar_video_sintetico(destino: Path, duracao_s: float = 5.0, fps: int = 24) -> Path:
    """Gera um video sintetico simples para testes.

    Args:
        destino: caminho de saida do `.mp4`.
        duracao_s: duracao em segundos.
        fps: frames por segundo.

    Returns:
        Caminho do arquivo gerado.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    largura, altura = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destino), fourcc, fps, (largura, altura))
    total_frames = int(duracao_s * fps)
    for i in range(total_frames):
        frame = np.full((altura, largura, 3), 30, dtype=np.uint8)
        x = int((i / total_frames) * (largura - 60))
        cv2.rectangle(frame, (x, 90), (x + 60, 150), (0, 200, 200), -1)
        writer.write(frame)
    writer.release()
    logger.info("Video sintetico gerado em %s (%d frames)", destino, total_frames)
    return destino


def imprimir_resumo(eventos: list[VideoEvent]) -> None:
    """Imprime um resumo dos eventos retornados pelo pipeline."""
    print(f"Total de eventos: {len(eventos)}")
    for evento in eventos[:3]:
        print(
            f"  frame={evento.frame_index:>4} "
            f"ts={evento.timestamp_ms:>6}ms "
            f"det={len(evento.detections):>2} "
            f"emocao={evento.facial_emotion.label if evento.facial_emotion else 'N/A':<8}"
        )
    if len(eventos) > 3:
        print(f"  ... ({len(eventos) - 3} eventos a mais)")


def main() -> None:
    if len(sys.argv) >= 2:
        video_path = Path(sys.argv[1])
    else:
        video_path = DEFAULT_DEMO_PATH
        if not video_path.exists():
            gerar_video_sintetico(video_path)

    pipeline = VideoPipeline(target_fps=1.0)
    eventos = pipeline.process(video_path)
    imprimir_resumo(eventos)
    if not eventos:
        logger.warning("Pipeline retornou lista vazia")
        sys.exit(1)
    if not all(isinstance(e, VideoEvent) for e in eventos):
        logger.error("Pipeline retornou item nao-VideoEvent")
        sys.exit(2)


if __name__ == "__main__":
    main()
