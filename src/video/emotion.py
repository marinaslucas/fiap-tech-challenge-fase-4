"""Detector de emocao facial.

Oferece dois backends:

- `FacialEmotionDetector` (FER local): usa `justinshenk/fer`. Exige
  `Pillow` legado (nao builda em Python 3.14). Em ambiente sem suporte,
  retorna lista vazia em vez de levantar excecao.
- `AzureOpenAIVisionEmotion` (cloud): usa GPT-4o vision no Azure AI
  Foundry, sem o vies de FER-2013 (que rotula faces femininas em repouso
  como `angry`/`sad`). Ativado quando `AZURE_OPENAI_VISION_DEPLOYMENT`
  esta preenchido no `.env`.

`get_facial_emotion_classifier()` seleciona o backend disponivel.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from src.config.settings import settings
from src.video.types import BoundingBox, EmotionScore

logger = logging.getLogger(__name__)


class FacialEmotionClassifierProtocol(Protocol):
    """Interface comum para classificadores de emocao facial.

    Tanto `FacialEmotionDetector` (FER local) quanto
    `AzureOpenAIVisionEmotion` (cloud) implementam essa assinatura,
    permitindo trocas plug-and-play no `VideoPipeline`.
    """

    def classify(self, image: object) -> EmotionScore | None:
        """Classifica a emocao predominante. Retorna `None` em fallback."""
        ...


class FacialEmotionDetector:
    """Classificador de emocao facial usando FER.

    O modelo e carregado lazy. Se a importacao do FER falhar (ambiente
    sem suporte), o detector marca-se como indisponivel e retorna [].
    """

    def __init__(self, use_mtcnn: bool = False) -> None:
        """Configura o detector.

        Args:
            use_mtcnn: usa MTCNN para deteccao de face (mais preciso, mais lento).
                Default usa o detector OpenCV Haar cascade do FER (mais rapido).
        """
        self.use_mtcnn: bool = use_mtcnn
        self._detector = None
        self._available: bool = False
        self._load_attempted: bool = False

    def load(self) -> None:
        """Carrega o FER. Em caso de falha, marca-se indisponivel."""
        self._load_attempted = True
        try:
            from fer import FER

            self._detector = FER(mtcnn=self.use_mtcnn)
            self._available = True
            logger.info("FER carregado (mtcnn=%s)", self.use_mtcnn)
        except ImportError as exc:
            logger.warning(
                "FER indisponivel no ambiente atual (%s). "
                "Detector facial retornara lista vazia ate ambiente compativel.",
                exc,
            )
            self._available = False

    def detect(self, frame: np.ndarray) -> list[EmotionScore]:
        """Detecta faces e classifica emocoes em um frame BGR.

        Args:
            frame: imagem BGR (HxWx3).

        Returns:
            Lista de `EmotionScore`, uma por face encontrada. Lista vazia
            se nenhuma face for detectada ou se o FER estiver indisponivel.
        """
        if not self._load_attempted:
            self.load()
        if not self._available or self._detector is None:
            return []
        results = self._detector.detect_emotions(frame)
        scores: list[EmotionScore] = []
        for entry in results:
            scores.append(_parse_fer_entry(entry))
        return scores

    def classify(self, image: np.ndarray) -> EmotionScore | None:
        """Implementa `FacialEmotionClassifierProtocol`: retorna a primeira face detectada.

        Wrapper sobre `detect()` para compatibilidade de interface com
        `AzureOpenAIVisionEmotion`. Retorna `None` quando nenhuma face
        e encontrada ou o FER nao esta disponivel.

        Args:
            image: frame BGR (HxWx3).

        Returns:
            `EmotionScore` da primeira face ou `None`.
        """
        scores = self.detect(image)
        return scores[0] if scores else None


def _parse_fer_entry(entry: dict) -> EmotionScore:
    """Converte dicionario do FER em `EmotionScore`."""
    emotions = entry.get("emotions", {})
    if not emotions:
        return EmotionScore(label="unknown", confidence=0.0, scores={})
    label, confidence = max(emotions.items(), key=lambda kv: kv[1])
    bbox = None
    box = entry.get("box")
    if box is not None:
        x, y, w, h = box
        bbox = BoundingBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h))
    return EmotionScore(
        bbox=bbox,
        label=str(label),
        confidence=float(confidence),
        scores={k: float(v) for k, v in emotions.items()},
    )


def get_facial_emotion_classifier() -> FacialEmotionClassifierProtocol:
    """Seleciona o classificador de emocao facial conforme `.env`.

    Quando `AZURE_OPENAI_VISION_DEPLOYMENT` esta preenchido E o cliente
    Azure consegue inicializar, retorna `AzureOpenAIVisionEmotion`
    (cloud, sem vies de FER-2013). Caso contrario, cai para
    `FacialEmotionDetector` (FER local). A decisao e lazy.
    """
    if settings.azure_openai_vision_deployment:
        from src.video.azure_openai_vision import AzureOpenAIVisionEmotion

        cloud = AzureOpenAIVisionEmotion()
        if cloud.is_configured:
            logger.info(
                "Usando AzureOpenAIVisionEmotion (deployment=%s)",
                cloud.deployment,
            )
            return cloud
        logger.warning(
            "AZURE_OPENAI_VISION_DEPLOYMENT preenchido mas Azure OpenAI nao "
            "configurado completamente; caindo no FER local."
        )
    return FacialEmotionDetector()
