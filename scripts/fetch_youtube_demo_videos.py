"""Baixa videos demo do YouTube para data/examples/<caso>/video.mp4.

Le `scripts/configs/demo_videos.yml` com URL + segmento (start, duration)
por caso. Usa `yt-dlp` para baixar e `ffmpeg` para trim. Opcionalmente
extrai o audio em PCM 16 kHz mono (formato esperado pelo Whisper) quando
`extract_audio: true`.

Uso:
    python scripts/fetch_youtube_demo_videos.py                  # todos os casos
    python scripts/fetch_youtube_demo_videos.py --case consulta_clinica_geral  # so 1
    python scripts/fetch_youtube_demo_videos.py --force          # refaz mesmo se existe

Pre-requisitos:
    yt-dlp (pip install yt-dlp ou brew install yt-dlp)
    ffmpeg (brew install ffmpeg)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "configs" / "demo_videos.yml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_youtube_demo_videos")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Caminho do YAML de config (default: {CONFIG_PATH.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument("--case", type=str, help="Roda so o caso especificado.")
    parser.add_argument(
        "--force", action="store_true", help="Re-baixa mesmo se ja existir."
    )
    return parser.parse_args()


def _check_tool(name: str) -> None:
    """Confirma que um binario esta no PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(
            f"{name} nao encontrado no PATH. "
            f"Instalar antes (ex: brew install {name})."
        )


def _download_video(url: str, dst: Path) -> None:
    """Baixa o video completo do YouTube via yt-dlp."""
    if dst.exists():
        dst.unlink()
    subprocess.run(
        [
            "yt-dlp",
            "--format",
            "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            "--no-playlist",
            "-o",
            str(dst),
            url,
        ],
        check=True,
    )


def _trim_video(src: Path, dst: Path, start: str, duration: int) -> None:
    """Recorta um segmento do video original via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(src),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _extract_audio(video: Path, audio_out: Path) -> None:
    """Extrai audio do video em PCM 16 kHz mono (formato Whisper)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def process_case(case_name: str, cfg: dict, force: bool = False) -> None:
    """Pipeline completo para um caso: download + trim + (opcional) audio.

    O caso vai pra `data/examples/<category>/<case_name>/` se a chave
    `category` estiver no config, ou `data/examples/<case_name>/` se ausente.
    """
    category = cfg.get("category", "").strip()
    case_dir = EXAMPLES_DIR / category / case_name if category else EXAMPLES_DIR / case_name
    video_out = case_dir / "video.mp4"

    if video_out.exists() and not force:
        rel = video_out.relative_to(PROJECT_ROOT)
        logger.info("[skip] %s ja existe (use --force pra refazer)", rel)
        return

    url = cfg.get("url", "").strip()
    if not url:
        logger.info(
            "[skip] %s sem URL (caso gerado por outro script, ex: build_cirurgia_demo_video.py)",
            case_name,
        )
        return

    start = cfg.get("start", "00:00:00")
    duration = cfg.get("duration", 30)
    extract_audio = cfg.get("extract_audio", False)
    discard_video = cfg.get("discard_video_after_extract", False)

    case_dir.mkdir(parents=True, exist_ok=True)
    tmp_full = case_dir / "_tmp_full.mp4"

    try:
        logger.info("[%s] baixando %s ...", case_name, url)
        _download_video(url, tmp_full)

        logger.info(
            "[%s] recortando start=%s duration=%ss -> %s",
            case_name,
            start,
            duration,
            video_out.name,
        )
        _trim_video(tmp_full, video_out, start, duration)

        if extract_audio:
            audio_out = case_dir / "audio.wav"
            logger.info("[%s] extraindo audio -> %s", case_name, audio_out.name)
            _extract_audio(video_out, audio_out)
    finally:
        if tmp_full.exists():
            tmp_full.unlink()

    if discard_video and video_out.exists():
        video_out.unlink()
        logger.info("[%s] video.mp4 descartado (audio-only por config)", case_name)
        return

    size_mb = video_out.stat().st_size / (1024 * 1024)
    logger.info(
        "[%s] OK -> %s (%.1f MB)",
        case_name,
        video_out.relative_to(PROJECT_ROOT),
        size_mb,
    )


def main() -> int:
    """Le config, valida ferramentas, processa cada caso."""
    args = _parse_args()

    _check_tool("yt-dlp")
    _check_tool("ffmpeg")

    if not args.config.exists():
        logger.error("Config nao encontrado: %s", args.config)
        return 1

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "cases" not in config:
        logger.error("Config invalido: chave 'cases' esperada")
        return 1

    cases = config["cases"]
    if args.case:
        if args.case not in cases:
            logger.error(
                "Caso '%s' nao encontrado. Disponiveis: %s",
                args.case,
                ", ".join(cases.keys()),
            )
            return 1
        cases = {args.case: cases[args.case]}

    failures: list[str] = []
    for case_name, cfg in cases.items():
        try:
            process_case(case_name, cfg, force=args.force)
        except Exception as exc:  # noqa: BLE001 - reportar mas seguir
            logger.error("[%s] falhou: %s", case_name, exc)
            failures.append(case_name)

    if failures:
        logger.error("Falhas em: %s", ", ".join(failures))
        return 1
    logger.info("Todos os casos processados com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
