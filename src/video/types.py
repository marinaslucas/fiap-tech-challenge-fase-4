"""Tipos compartilhados do pipeline de video.

Modelos Pydantic v2 usados por `detector.py`, `emotion.py` e agregados
em `pipeline.py`. Manter aqui para evitar imports circulares.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Caixa delimitadora em coordenadas absolutas de pixel.

    Usa convencao xyxy (canto superior esquerdo, canto inferior direito).
    """

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Largura em pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Altura em pixels."""
        return self.y2 - self.y1


class Detection(BaseModel):
    """Uma deteccao de objeto produzida pelo detector YOLO."""

    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class EmotionScore(BaseModel):
    """Resultado de classificacao de emocao para uma face.

    `scores` traz a distribuicao completa de probabilidades. `label` e a
    emocao predominante e `confidence` e a probabilidade dela.

    `body_language` (opcional) traz a categoria predominante de linguagem
    corporal interpretada pelo classifier de visao multimodal (tranquila,
    tensa, retraida, agitada, indefinida). Backends que nao oferecem o
    sinal (FER local) deixam `None`.
    """

    bbox: BoundingBox | None = None
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)
    body_language: str | None = None

    @property
    def label_pt(self) -> str:
        """Versao em PT-BR do label (ver `src.audio.types.EMOTION_LABEL_PT`)."""
        from src.audio.types import EMOTION_LABEL_PT

        return EMOTION_LABEL_PT.get(self.label.lower(), self.label)


class VideoEvent(BaseModel):
    """Saida agregada por frame do pipeline de video.

    Cada `VideoEvent` representa um frame amostrado pelo pipeline,
    contendo todas as analises feitas naquele instante.
    """

    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    detections: list[Detection] = Field(default_factory=list)
    facial_emotion: EmotionScore | None = None
