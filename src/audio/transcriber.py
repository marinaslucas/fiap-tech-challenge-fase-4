"""Transcricao de audio com toggle local (faster-whisper) ou cloud (Azure Speech).

Por padrao usa `faster-whisper` localmente. Quando `USE_CLOUD_TRANSCRIPTION=true`
no `.env` E o cliente Azure Speech esta configurado, usa Azure como provider.
Em qualquer caso a saida segue a mesma interface `(text, segments)`, o que mantem
o pipeline a jusante agnostico ao provider.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from src.audio.types import Segment
from src.config.settings import settings

logger = logging.getLogger(__name__)


class TranscriberProtocol(Protocol):
    """Interface comum: recebe um arquivo de audio e devolve `(texto, segmentos)`."""

    def transcribe(self, audio_path: Path) -> tuple[str, list[Segment]]: ...


class WhisperTranscriber:
    """Transcricao local via `faster-whisper`.

    Lazy-load do modelo para evitar custo de boot quando o transcriber
    nao chega a ser usado.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str = "pt",
        compute_type: str = "int8",
    ) -> None:
        """Configura o transcriber local.

        Args:
            model_size: tamanho do Whisper (`tiny`, `base`, `small`, `medium`, `large`).
                `small` equilibra qualidade e tempo em CPU.
            language: codigo ISO de idioma. PT-BR usa `pt`.
            compute_type: precisao de inferencia. `int8` e mais rapido em CPU
                com perda minima de qualidade.
        """
        self.model_size: str = model_size
        self.language: str = language
        self.compute_type: str = compute_type
        self._model = None
        self._available: bool = False
        self._load_attempted: bool = False

    def load(self) -> None:
        """Carrega o modelo Whisper. Marca-se indisponivel em caso de falha."""
        self._load_attempted = True
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
            )
            self._available = True
            logger.info(
                "faster-whisper carregado (model=%s, compute_type=%s)",
                self.model_size,
                self.compute_type,
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning(
                "faster-whisper indisponivel (%s). Transcritor retornara texto vazio.",
                exc,
            )
            self._available = False

    def transcribe(self, audio_path: Path) -> tuple[str, list[Segment]]:
        """Transcreve um arquivo de audio.

        Args:
            audio_path: caminho do arquivo (`.wav`, `.mp3`, etc.).

        Returns:
            Tupla `(texto_completo, lista_de_segmentos)`. Em caso de falha
            de carregamento, retorna `("", [])`.
        """
        if not self._load_attempted:
            self.load()
        if not self._available or self._model is None:
            return "", []
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=1,
            vad_filter=True,
        )
        segments: list[Segment] = []
        chunks: list[str] = []
        for seg in segments_iter:
            segments.append(
                Segment(
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                    text=seg.text.strip(),
                )
            )
            chunks.append(seg.text.strip())
        text = " ".join(chunk for chunk in chunks if chunk)
        logger.info(
            "Whisper concluiu: lang=%s, duracao=%.2fs, segmentos=%d",
            info.language,
            info.duration,
            len(segments),
        )
        return text, segments


class AzureSpeechTranscriber:
    """Transcricao cloud via Azure Speech com reconhecimento continuo.

    Diferente de `recognize_once_async` (que para na primeira pausa),
    o reconhecimento continuo processa o audio todo, acumulando frases
    em segmentos com timestamps. Cada `Recognized` event vira um
    `Segment`, e o texto final e a concatenacao deles.
    """

    def __init__(self, language: str = "pt-BR") -> None:
        """Configura o cliente Azure Speech.

        Args:
            language: codigo BCP-47 do idioma (default PT-BR).
        """
        self.language: str = language
        self.api_key: str = settings.azure_speech_key.get_secret_value()
        self.region: str = settings.azure_speech_region

    @property
    def is_configured(self) -> bool:
        """`True` se a chave Azure Speech esta presente no `.env`."""
        return bool(self.api_key and self.region)

    def transcribe(self, audio_path: Path) -> tuple[str, list[Segment]]:
        """Transcreve audio inteiro via Azure Speech continuous recognition.

        Returns:
            `(texto, segmentos)` ou `("", [])` se o cliente nao estiver configurado.
        """
        if not self.is_configured:
            logger.info("Azure Speech nao configurado; pulando transcricao cloud")
            return "", []
        import threading

        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(subscription=self.api_key, region=self.region)
        speech_config.speech_recognition_language = self.language
        audio_input = speechsdk.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_input
        )

        segments: list[Segment] = []
        done = threading.Event()

        def on_recognized(evt) -> None:
            """Chamado quando o Azure devolve uma frase finalizada."""
            if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
                return
            text = evt.result.text.strip()
            if not text:
                return
            # offset/duration vem em 100-nanosegundos (ticks). Converte pra ms.
            start_ms = int(evt.result.offset / 10000)
            end_ms = start_ms + int(evt.result.duration / 10000)
            segments.append(Segment(start_ms=start_ms, end_ms=end_ms, text=text))

        def on_session_stopped(_evt) -> None:
            """Chamado quando o reconhecimento termina (fim do audio ou cancelamento)."""
            done.set()

        def on_canceled(evt) -> None:
            """Chamado em erro ou cancelamento pelo servico."""
            if evt.reason == speechsdk.CancellationReason.Error:
                logger.warning(
                    "Azure Speech cancelado por erro: %s | %s",
                    evt.error_code, evt.error_details,
                )
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(on_session_stopped)
        recognizer.canceled.connect(on_canceled)

        recognizer.start_continuous_recognition_async().get()
        done.wait()
        recognizer.stop_continuous_recognition_async().get()

        text = " ".join(s.text for s in segments)
        logger.info(
            "Azure Speech transcreveu %d caracteres em %d segmento(s)",
            len(text), len(segments),
        )
        return text, segments


def get_transcriber() -> TranscriberProtocol:
    """Seleciona o transcriber conforme `USE_CLOUD_TRANSCRIPTION` no `.env`.

    Returns:
        `AzureSpeechTranscriber` se o toggle estiver `true` E o cliente
        estiver configurado; caso contrario, `WhisperTranscriber` local.
    """
    if settings.use_cloud_transcription:
        cloud = AzureSpeechTranscriber()
        if cloud.is_configured:
            logger.info("Usando AzureSpeechTranscriber (toggle ativo)")
            return cloud
        logger.warning(
            "USE_CLOUD_TRANSCRIPTION=true mas Azure Speech nao configurado; "
            "voltando para Whisper local"
        )
    return WhisperTranscriber()
