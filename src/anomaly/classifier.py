"""Classificador final que combina regras clinicas e detector estatistico.

Encapsula a heuristica de fusao entre `rules.evaluate_rules` e
`statistical.StatisticalAnomalyDetector`. Producoes:

- Triggers de regras tem prioridade. Eles entram primeiro na lista final
- Se o detector estatistico estiver treinado e pontuar acima do limiar,
  um trigger `statistical.outlier` e adicionado em nivel `moderate`
- O nivel final e o maior entre o nivel das regras e o nivel estatistico
- Triggers `critical` nao sao rebaixados; apenas elevados (rules-first)

Quando o detector estatistico nao esta disponivel ou nao foi treinado,
o classificador degrada para o resultado puro das regras. Esse padrao
mantem o pipeline operacional desde a primeira execucao.
"""

from __future__ import annotations

import logging

from src.anomaly.rules import (
    derive_risk_level,
    evaluate_rules,
    recommend_actions,
)
from src.anomaly.statistical import (
    DEFAULT_SCORE_THRESHOLD,
    StatisticalAnomalyDetector,
    extract_feature_vector,
)
from src.anomaly.types import AnomalyResult, Trigger
from src.audio.types import AudioAnalysis
from src.video.types import VideoEvent

logger = logging.getLogger(__name__)


class AnomalyClassifier:
    """Classificador hibrido de anomalia.

    Mantem opcionalmente um `StatisticalAnomalyDetector` injetado. Quando
    nao recebe um, o classificador opera apenas com regras (modo seguro).
    """

    def __init__(
        self,
        statistical_detector: StatisticalAnomalyDetector | None = None,
        statistical_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        """Configura o classificador.

        Args:
            statistical_detector: detector estatistico ja treinado (ou nao).
                Quando `None`, o classificador opera so com regras.
            statistical_threshold: score >= esse valor gera trigger.
        """
        self.statistical_detector: StatisticalAnomalyDetector | None = statistical_detector
        self.statistical_threshold: float = statistical_threshold

    def classify(
        self,
        video_events: list[VideoEvent] | None = None,
        audio_analysis: AudioAnalysis | None = None,
    ) -> AnomalyResult:
        """Aplica regras + detector estatistico e devolve `AnomalyResult`.

        Args:
            video_events: lista de eventos do pipeline de video, opcional.
            audio_analysis: saida do pipeline de audio, opcional.

        Returns:
            `AnomalyResult` consolidado, com triggers de regras primeiro e,
            quando aplicavel, um trigger estatistico ao final.
        """
        triggers: list[Trigger] = evaluate_rules(video_events, audio_analysis)

        statistical_trigger = self._maybe_statistical_trigger(video_events, audio_analysis)
        if statistical_trigger is not None:
            triggers.append(statistical_trigger)

        level = derive_risk_level(triggers)
        actions = recommend_actions(triggers)
        explanation = self._build_explanation(triggers)

        logger.info(
            "Classificacao concluida: nivel=%s, %d trigger(s).",
            level,
            len(triggers),
        )
        return AnomalyResult(
            level=level,
            triggers=triggers,
            explanation=explanation,
            recommended_actions=actions,
        )

    def _maybe_statistical_trigger(
        self,
        video_events: list[VideoEvent] | None,
        audio_analysis: AudioAnalysis | None,
    ) -> Trigger | None:
        """Gera trigger estatistico se o detector estiver treinado e pontuar alto."""
        detector = self.statistical_detector
        if detector is None or not detector.is_fitted:
            return None
        features = extract_feature_vector(video_events, audio_analysis)
        score = detector.score(features)
        if score is None or score < self.statistical_threshold:
            return None
        return Trigger(
            rule_id="statistical.outlier",
            level="moderate",
            message=(
                f"Caso classificado como outlier pelo Isolation Forest"
                f" (score={score:.2f} >= {self.statistical_threshold:.2f})."
            ),
            source="fusion",
            evidence={
                "score": round(score, 3),
                "threshold": self.statistical_threshold,
            },
        )

    def _build_explanation(self, triggers: list[Trigger]) -> str:
        """Concatena as mensagens dos triggers em uma linha auditavel."""
        if not triggers:
            return "Nenhuma anomalia detectada por regras ou detector estatistico."
        return "; ".join(t.message for t in triggers)
