"""Deteccao de anomalias: regras clinicas, statistical e classificador final."""

from src.anomaly.classifier import AnomalyClassifier
from src.anomaly.rules import (
    AUDIO_RULES,
    VIDEO_RULES,
    build_anomaly_result,
    derive_risk_level,
    evaluate_rules,
    recommend_actions,
)
from src.anomaly.statistical import (
    DEFAULT_SCORE_THRESHOLD,
    FEATURE_NAMES,
    StatisticalAnomalyDetector,
    extract_feature_vector,
    features_to_array,
)
from src.anomaly.types import AnomalyResult, RiskLevel, Trigger

__all__ = [
    "AUDIO_RULES",
    "AnomalyClassifier",
    "AnomalyResult",
    "DEFAULT_SCORE_THRESHOLD",
    "FEATURE_NAMES",
    "RiskLevel",
    "StatisticalAnomalyDetector",
    "Trigger",
    "VIDEO_RULES",
    "build_anomaly_result",
    "derive_risk_level",
    "evaluate_rules",
    "extract_feature_vector",
    "features_to_array",
    "recommend_actions",
]
