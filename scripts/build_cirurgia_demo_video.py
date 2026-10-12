"""Compila video MP4 de cirurgia laparoscopica a partir do CholecSeg8k.

Modos suportados:

1. `--mode sequential` (default): pega N frames sequenciais a partir de
   `--start-idx`. Saida: cirurgia rotineira sem filtro de conteudo. Vai
   para `data/examples/cirurgias/rotina/video.mp4`.

2. `--mode blood`: usa o modelo YOLO v1 custom (`models/yolov8n_surgical.pt`)
   pra detectar a classe `blood` em cada frame do split, ranqueia por
   confianca e empacota os top N frames em ordem temporal original. Saida:
   `data/examples/cirurgias/sangramento/video.mp4`.

Pre-requisitos:
    pip install datasets pillow ultralytics
    ffmpeg (brew install ffmpeg)
    models/yolov8n_surgical.pt commitado no repo (modo blood)

Uso:
    python scripts/build_cirurgia_demo_video.py --mode sequential
    python scripts/build_cirurgia_demo_video.py --mode blood
    python scripts/build_cirurgia_demo_video.py --mode blood --n-frames 40 --fps 6
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n_surgical.pt"

OUT_DIR_BY_MODE = {
    "sequential": EXAMPLES_DIR / "cirurgias" / "rotina",
    "blood":      EXAMPLES_DIR / "cirurgias" / "sangramento",
}

# Classe do modelo v1 que indica sangramento (verificado: m.names[2]='blood').
BLOOD_CLASS_ID = 2
# Confianca minima para considerar deteccao de sangue significativa.
BLOOD_MIN_CONFIDENCE = 0.3

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_cirurgia_demo_video")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode",
        choices=("sequential", "blood"),
        default="blood",
        help=(
            "sequential = frames consecutivos sem filtro; "
            "blood = filtrar por deteccao YOLO da classe blood (default: blood)"
        ),
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=30,
        help="Numero de frames a stitchar (default: 30)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames por segundo no MP4 final (default: 5)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Indice do primeiro frame no test split (so modo sequential, default: 0)",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="minwoosun/CholecSeg8k",
        help="Dataset HuggingFace (default: minwoosun/CholecSeg8k)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split do dataset a usar (default: train; HF minwoosun/CholecSeg8k so tem 'train')",
    )
    parser.add_argument(
        "--blood-conf-min",
        type=float,
        default=BLOOD_MIN_CONFIDENCE,
        help=f"Confianca minima da deteccao blood (so modo blood, default: {BLOOD_MIN_CONFIDENCE})",
    )
    return parser.parse_args()


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg nao encontrado no PATH. Instalar com: brew install ffmpeg"
        )


def _select_blood_frames(ds, n_frames: int, conf_min: float) -> list[int]:
    """Roda v1 YOLO em cada frame, retorna indices dos top N por confianca blood."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo YOLO v1 nao encontrado em {MODEL_PATH}. "
            "Necessario pra modo blood."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics nao instalado. Instalar com: pip install ultralytics"
        ) from exc

    logger.info("Carregando modelo YOLO v1 de %s", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))
    logger.info("Classes do modelo: %s", model.names)
    if model.names.get(BLOOD_CLASS_ID) != "blood":
        raise RuntimeError(
            f"Esperava class id {BLOOD_CLASS_ID} = 'blood', "
            f"encontrei: {model.names}"
        )

    # (idx, max_blood_confidence)
    scores: list[tuple[int, float]] = []
    total = len(ds)
    log_every = max(1, total // 20)

    logger.info("Rodando inferencia em %d frames do split...", total)
    for i in range(total):
        img = ds[i]["image"]  # PIL
        result = model.predict(img, verbose=False)[0]
        max_blood = 0.0
        if result.boxes is not None and len(result.boxes) > 0:
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            blood_mask = cls == BLOOD_CLASS_ID
            if blood_mask.any():
                max_blood = float(conf[blood_mask].max())
        scores.append((i, max_blood))

        if (i + 1) % log_every == 0:
            n_above = sum(1 for _, c in scores if c >= conf_min)
            logger.info("  %d/%d frames | %d acima de conf %.2f",
                        i + 1, total, n_above, conf_min)

    # Filtra acima do threshold, ordena por confianca desc, pega top N
    candidatos = [(i, c) for i, c in scores if c >= conf_min]
    candidatos.sort(key=lambda x: x[1], reverse=True)
    selecionados = candidatos[:n_frames]
    if len(selecionados) < n_frames:
        logger.warning(
            "Apenas %d frames com blood >= %.2f (pediu %d). Vou usar todos.",
            len(selecionados), conf_min, n_frames,
        )

    # Reordena por indice original pra preservar coerencia temporal
    indices = sorted(idx for idx, _ in selecionados)
    logger.info(
        "Selecionados %d frames (confs %.2f a %.2f)",
        len(indices),
        min(c for _, c in selecionados) if selecionados else 0.0,
        max(c for _, c in selecionados) if selecionados else 0.0,
    )
    return indices


def main() -> int:
    args = _parse_args()
    _check_ffmpeg()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets nao instalado. Instalar com: pip install datasets pillow"
        ) from exc

    out_dir = OUT_DIR_BY_MODE[args.mode]
    out_video = out_dir / "video.mp4"

    logger.info("Carregando %s split=%s ...", args.hf_dataset, args.split)
    ds = load_dataset(args.hf_dataset, split=args.split)
    total = len(ds)
    logger.info("Dataset carregado: %d frames disponiveis", total)

    if args.mode == "sequential":
        end_idx = args.start_idx + args.n_frames
        if end_idx > total:
            raise ValueError(
                f"start_idx + n_frames ({end_idx}) excede tamanho do split ({total})"
            )
        indices = list(range(args.start_idx, end_idx))
    else:  # blood
        indices = _select_blood_frames(ds, args.n_frames, args.blood_conf_min)

    if not indices:
        raise RuntimeError("Nenhum frame selecionado pra stitchar")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"cirurgia_{args.mode}_"))
    try:
        logger.info("Salvando %d frames em %s", len(indices), tmp_dir)
        for out_idx, ds_idx in enumerate(indices):
            img = ds[ds_idx]["image"]
            img.save(tmp_dir / f"frame_{out_idx:04d}.png")

        logger.info("Empacotando em MP4 @ %d fps -> %s", args.fps, out_video.name)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(tmp_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(out_video),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        size_mb = out_video.stat().st_size / (1024 * 1024)
        duration_s = len(indices) / args.fps
        logger.info(
            "OK -> %s (%.1f MB, ~%.1fs)",
            out_video.relative_to(PROJECT_ROOT),
            size_mb,
            duration_s,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
