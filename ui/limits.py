"""Validacoes de input para a UI Gradio.

Centraliza limites de upload (tamanho e duracao) para evitar OOM ou
timeouts em videos/audios grandes enviados pelo usuario. Cada aba chama
`validate_video` / `validate_audio` antes de despachar pro pipeline; se
falhar, devolve mensagem clara ao inves de processar e travar.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Limites calibrados para a demo: input rapido (ate 2min) sem estourar memoria.
MAX_VIDEO_BYTES: int = 60 * 1024 * 1024  # 60 MB
MAX_AUDIO_BYTES: int = 30 * 1024 * 1024  # 30 MB
MAX_DURATION_SECONDS: float = 120.0


class ValidationResult(NamedTuple):
    """Resultado de uma validacao de upload."""

    ok: bool
    message: str


def _ffprobe_duration(path: Path) -> float | None:
    """Retorna a duracao do midia em segundos, ou `None` se nao for legivel."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        logger.debug("ffprobe falhou em %s: %s", path, exc)
        return None


def _format_mb(b: int) -> str:
    return f"{b / (1024 * 1024):.1f} MB"


def validate_video(path: Path) -> ValidationResult:
    """Confere tamanho e duracao do arquivo de video.

    Returns:
        `ValidationResult` com `ok=True` se passar dentro dos limites,
        senao `ok=False` com mensagem explicando como ajustar.
    """
    if not path.exists():
        return ValidationResult(False, "Arquivo de video nao encontrado.")

    size = path.stat().st_size
    if size > MAX_VIDEO_BYTES:
        return ValidationResult(
            False,
            f"Video muito grande: {_format_mb(size)}. "
            f"Limite: {MAX_VIDEO_BYTES // (1024 * 1024)} MB. "
            f"Cortar trecho menor (~30-60s) antes de enviar.",
        )

    duration = _ffprobe_duration(path)
    if duration is not None and duration > MAX_DURATION_SECONDS:
        return ValidationResult(
            False,
            f"Video muito longo: {duration:.0f}s. "
            f"Limite: {MAX_DURATION_SECONDS:.0f}s. "
            f"Cortar trecho menor antes de enviar.",
        )

    return ValidationResult(True, "")


def validate_audio(path: Path) -> ValidationResult:
    """Confere tamanho e duracao do arquivo de audio.

    Returns:
        `ValidationResult` com `ok=True` se passar dentro dos limites,
        senao `ok=False` com mensagem explicando como ajustar.
    """
    if not path.exists():
        return ValidationResult(False, "Arquivo de audio nao encontrado.")

    size = path.stat().st_size
    if size > MAX_AUDIO_BYTES:
        return ValidationResult(
            False,
            f"Audio muito grande: {_format_mb(size)}. "
            f"Limite: {MAX_AUDIO_BYTES // (1024 * 1024)} MB.",
        )

    duration = _ffprobe_duration(path)
    if duration is not None and duration > MAX_DURATION_SECONDS:
        return ValidationResult(
            False,
            f"Audio muito longo: {duration:.0f}s. "
            f"Limite: {MAX_DURATION_SECONDS:.0f}s.",
        )

    return ValidationResult(True, "")
