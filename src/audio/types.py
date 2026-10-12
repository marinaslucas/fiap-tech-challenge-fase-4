"""Tipos compartilhados do pipeline de audio.

Modelos Pydantic v2 usados por `transcriber.py`, `features.py`, `emotion.py`
e agregados em `pipeline.py` no `AudioAnalysis`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """Segmento de transcricao com timestamps em milissegundos."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str


class AcousticFeatures(BaseModel):
    """Features acusticas extraidas via librosa.

    `mfcc_means` agrega o vetor de 13 coeficientes (media ao longo do tempo).
    `jitter` e `shimmer` aproximam variabilidade fina de pitch e amplitude;
    sao bons proxies para tensao vocal.
    """

    duration_s: float = Field(ge=0.0)
    pitch_mean_hz: float
    pitch_std_hz: float
    energy_rms: float
    zero_crossing_rate: float
    jitter: float
    shimmer: float
    mfcc_means: list[float] = Field(default_factory=list)


class EmotionScore(BaseModel):
    """Resultado da classificacao de emocao a partir da voz."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def label_pt(self) -> str:
        """Versao em PT-BR do label, pronta pra exibir na UI / mensagens de regras.

        Cobre labels do wav2vec2-superb-er (`ang`, `hap`, `neu`, `sad`) e dos
        modelos faciais FER / GPT-vision (`angry`, `happy`, `neutral`, etc.).
        Caso o label nao esteja no mapa, devolve o original.
        """
        return EMOTION_LABEL_PT.get(self.label.lower(), self.label)


# Mapa de codigos do modelo -> rotulo PT-BR amigavel.
# Usar `EmotionScore.label_pt` em vez de acessar este dict direto, exceto no
# pipeline de regras que pode receber o label como string isolada.
EMOTION_LABEL_PT: dict[str, str] = {
    # wav2vec2 superb-er (IEMOCAP 4 classes)
    "ang": "raiva",
    "hap": "alegria",
    "neu": "neutro",
    "sad": "tristeza",
    "fea": "medo",
    # FER local + GPT-vision (rotulos extendidos)
    "angry": "raiva",
    "happy": "alegria",
    "neutral": "neutro",
    "fearful": "medo",
    "fear": "medo",
    "disgust": "repulsa",
    "surprise": "surpresa",
}


class SentimentResult(BaseModel):
    """Sentimento da transcricao (output do Azure Language Sentiment Analysis)."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)


class AudioAnalysis(BaseModel):
    """Saida agregada do pipeline de audio para um arquivo completo."""

    transcription: str = ""
    segments: list[Segment] = Field(default_factory=list)
    acoustic_features: AcousticFeatures | None = None
    emotion: EmotionScore | None = None
    sentiment: SentimentResult | None = None
    key_phrases: list[str] = Field(default_factory=list)
