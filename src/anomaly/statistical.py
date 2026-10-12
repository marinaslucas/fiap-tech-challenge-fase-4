"""Detector estatistico de anomalia baseado em Isolation Forest.

Pega features agregadas dos pipelines de video e audio, monta um vetor
numerico e usa um Isolation Forest do scikit-learn para identificar
casos fora da distribuicao baseline.

Conceito de uso:
    1. Treinar offline com casos rotulados como "normais" (`fit`)
    2. Em producao, chamar `score(features)` para obter um valor [0, 1]
       onde valores mais altos indicam mais anomalia
    3. O classificador final (`classifier.py`) combina este score com as
       regras explicitas (`rules.py`).

Sem `fit` previo, o detector reporta `is_fitted=False` e `score` retorna
`None`, permitindo que o pipeline continue funcionando com apenas regras.
Esse padrao deliberadamente espelha o `is_configured` dos clientes cloud.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Iterable

from src.anomaly.rules import FACIAL_DISTRESS_LABELS, VOCAL_DISTRESS_LABELS
from src.audio.types import AudioAnalysis
from src.video.types import VideoEvent

logger = logging.getLogger(__name__)

# Ordem fixa das features no vetor numerico. Manter estavel para que modelos
# salvos sigam compativeis ao serem recarregados.
FEATURE_NAMES: tuple[str, ...] = (
    "audio.pitch_mean_hz",
    "audio.pitch_std_hz",
    "audio.energy_rms",
    "audio.zero_crossing_rate",
    "audio.jitter",
    "audio.shimmer",
    "audio.duration_s",
    "audio.emotion_distress_confidence",
    "audio.sentiment_negative_confidence",
    "audio.key_phrase_count",
    "video.frame_count",
    "video.avg_detections_per_frame",
    "video.surgical_instrument_frame_ratio",
    "video.facial_distress_ratio",
    "video.facial_emotion_confidence_mean",
)

# Score de saida normalizado em [0, 1]. Acima desse limiar o classificador
# (4.3) considera anomalia estatistica relevante.
DEFAULT_SCORE_THRESHOLD: float = 0.6


def extract_feature_vector(
    video_events: list[VideoEvent] | None = None,
    audio_analysis: AudioAnalysis | None = None,
) -> dict[str, float]:
    """Extrai um dicionario de features numericas para o detector.

    O dict retornado mapeia nomes em `FEATURE_NAMES` para valores `float`.
    Modalidades ausentes geram zeros, mantendo o vetor sempre completo.

    Args:
        video_events: lista de `VideoEvent` ou `None`.
        audio_analysis: `AudioAnalysis` ou `None`.

    Returns:
        Dict com todas as chaves de `FEATURE_NAMES` preenchidas com float.
    """
    features: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

    if audio_analysis is not None:
        acoustic = audio_analysis.acoustic_features
        if acoustic is not None:
            features["audio.pitch_mean_hz"] = float(acoustic.pitch_mean_hz)
            features["audio.pitch_std_hz"] = float(acoustic.pitch_std_hz)
            features["audio.energy_rms"] = float(acoustic.energy_rms)
            features["audio.zero_crossing_rate"] = float(acoustic.zero_crossing_rate)
            features["audio.jitter"] = float(acoustic.jitter)
            features["audio.shimmer"] = float(acoustic.shimmer)
            features["audio.duration_s"] = float(acoustic.duration_s)

        emotion = audio_analysis.emotion
        if emotion is not None and emotion.label.lower() in VOCAL_DISTRESS_LABELS:
            features["audio.emotion_distress_confidence"] = float(emotion.confidence)

        sentiment = audio_analysis.sentiment
        if sentiment is not None and sentiment.label.lower() == "negative":
            features["audio.sentiment_negative_confidence"] = float(sentiment.confidence)

        features["audio.key_phrase_count"] = float(len(audio_analysis.key_phrases))

    if video_events:
        features["video.frame_count"] = float(len(video_events))
        detection_counts = [len(event.detections) for event in video_events]
        features["video.avg_detections_per_frame"] = (
            statistics.fmean(detection_counts) if detection_counts else 0.0
        )

        instrument_frames = sum(
            1
            for event in video_events
            if any(
                det.class_name.lower() in {"grasper", "l_hook_electrocautery", "hook"}
                for det in event.detections
            )
        )
        features["video.surgical_instrument_frame_ratio"] = instrument_frames / len(video_events)

        facial_emotions = [e.facial_emotion for e in video_events if e.facial_emotion]
        if facial_emotions:
            distress_count = sum(
                1 for e in facial_emotions if e.label.lower() in FACIAL_DISTRESS_LABELS
            )
            features["video.facial_distress_ratio"] = distress_count / len(facial_emotions)
            features["video.facial_emotion_confidence_mean"] = statistics.fmean(
                e.confidence for e in facial_emotions
            )

    return features


def features_to_array(features: dict[str, float]) -> list[float]:
    """Converte um dict de features para lista na ordem canonica.

    Garante que o vetor sempre tenha o mesmo comprimento e ordem,
    independente da ordem das chaves no dict de entrada.
    """
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


class StatisticalAnomalyDetector:
    """Wrapper de Isolation Forest para detectar anomalia em features agregadas.

    A classe e independente das regras: opera sobre vetores numericos
    produzidos por `extract_feature_vector` + `features_to_array`. O modelo
    e treinado offline com casos baseline e usado em inferencia para
    pontuar novos casos.

    Score de saida e mapeado para o intervalo [0, 1] usando uma sigmoid
    centrada em 0 sobre o `decision_function` do sklearn (decision negativo
    indica anomalia, positivo indica inlier).
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
        n_estimators: int = 100,
    ) -> None:
        """Configura hiperparametros do Isolation Forest.

        Args:
            contamination: fracao esperada de anomalias no conjunto de treino
                (passada para o sklearn).
            random_state: semente para reprodutibilidade.
            n_estimators: numero de arvores do ensemble.
        """
        self.contamination: float = contamination
        self.random_state: int = random_state
        self.n_estimators: int = n_estimators
        self._model = None
        self._available: bool = False
        self._fitted: bool = False

    @property
    def is_available(self) -> bool:
        """`True` se o sklearn carregou (a classe pode ser instanciada)."""
        self._ensure_model()
        return self._available

    @property
    def is_fitted(self) -> bool:
        """`True` se `fit` ja foi chamado com sucesso."""
        return self._fitted

    def _ensure_model(self) -> None:
        """Lazy-load do `IsolationForest`. Marca-se indisponivel se sklearn ausente."""
        if self._model is not None or self._available:
            return
        try:
            from sklearn.ensemble import IsolationForest

            self._model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=self.n_estimators,
            )
            self._available = True
            logger.info("Isolation Forest pronto (contamination=%.2f)", self.contamination)
        except ImportError as exc:
            logger.warning(
                "scikit-learn nao disponivel, detector estatistico desativado: %s",
                exc,
            )
            self._available = False

    def fit(self, samples: Iterable[dict[str, float]]) -> None:
        """Treina o detector com casos baseline.

        Args:
            samples: iteravel de dicts de features (saida de
                `extract_feature_vector`). Cada amostra deve ter as chaves
                de `FEATURE_NAMES`; chaves faltantes recebem 0.

        Raises:
            ValueError: se nenhuma amostra for fornecida.
        """
        self._ensure_model()
        if not self._available or self._model is None:
            return
        data = [features_to_array(sample) for sample in samples]
        if not data:
            raise ValueError("Pelo menos uma amostra e necessaria para treinar.")
        self._model.fit(data)
        self._fitted = True
        logger.info("Detector treinado com %d amostras.", len(data))

    def score(self, features: dict[str, float]) -> float | None:
        """Calcula score normalizado de anomalia em [0, 1].

        Args:
            features: dict de features de um caso.

        Returns:
            Score em [0, 1] (mais alto = mais anomalo), ou `None` se o
            detector nao estiver treinado ou disponivel.
        """
        self._ensure_model()
        if not self._available or self._model is None or not self._fitted:
            return None
        vector = [features_to_array(features)]
        # decision_function: positivo = inlier, negativo = outlier.
        raw = float(self._model.decision_function(vector)[0])
        # Sigmoid invertida: anomalo proximo de 1.0 quando `raw` e negativo.
        normalized = 1.0 / (1.0 + math.exp(raw * 4))
        return max(0.0, min(1.0, normalized))

    def is_anomalous(
        self,
        features: dict[str, float],
        threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> bool:
        """Atalho que devolve `True` se `score(features) >= threshold`."""
        value = self.score(features)
        return value is not None and value >= threshold
