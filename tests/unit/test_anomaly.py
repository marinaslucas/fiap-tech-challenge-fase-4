"""Testes smoke das regras clinicas de anomalia.

Cobrem cada regra com pelo menos um caso positivo e um negativo,
mais um teste do agregador `evaluate_rules` e do atalho
`build_anomaly_result`. Sem dependencia de modelo ou rede: usam
diretamente os tipos Pydantic dos pipelines.
"""

from __future__ import annotations

import pytest

from src.anomaly import (
    FEATURE_NAMES,
    AnomalyClassifier,
    AnomalyResult,
    StatisticalAnomalyDetector,
    Trigger,
    build_anomaly_result,
    derive_risk_level,
    evaluate_rules,
    extract_feature_vector,
    features_to_array,
    recommend_actions,
)
from src.anomaly.rules import (
    SURGICAL_INSTRUMENT_PRESENCE_THRESHOLD,
    rule_critical_terms,
    rule_facial_distress,
    rule_low_vocal_energy,
    rule_negative_sentiment,
    rule_surgical_instrument_presence,
    rule_vocal_distress,
    rule_vocal_strain,
)
from src.audio.types import (
    AcousticFeatures,
    AudioAnalysis,
    SentimentResult,
)
from src.audio.types import (
    EmotionScore as AudioEmotionScore,
)
from src.video.types import (
    BoundingBox,
    Detection,
    VideoEvent,
)
from src.video.types import (
    EmotionScore as VideoEmotionScore,
)

# --------------------------------------------------------------------------
# Helpers de construcao
# --------------------------------------------------------------------------


def _bbox() -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=10, y2=10)


def _event(
    frame_index: int,
    *,
    instrument: bool = False,
    instrument_conf: float = 0.8,
    facial_label: str | None = None,
    facial_conf: float = 0.9,
) -> VideoEvent:
    detections = []
    if instrument:
        detections.append(
            Detection(
                class_id=0,
                class_name="grasper",
                confidence=instrument_conf,
                bbox=_bbox(),
            )
        )
    facial = None
    if facial_label is not None:
        facial = VideoEmotionScore(label=facial_label, confidence=facial_conf, bbox=_bbox())
    return VideoEvent(
        frame_index=frame_index,
        timestamp_ms=frame_index * 1000,
        detections=detections,
        facial_emotion=facial,
    )


def _audio(
    *,
    transcription: str = "",
    emotion: AudioEmotionScore | None = None,
    features: AcousticFeatures | None = None,
    sentiment: SentimentResult | None = None,
    key_phrases: list[str] | None = None,
) -> AudioAnalysis:
    return AudioAnalysis(
        transcription=transcription,
        segments=[],
        acoustic_features=features,
        emotion=emotion,
        sentiment=sentiment,
        key_phrases=key_phrases or [],
    )


# --------------------------------------------------------------------------
# Regras de video
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_rule_surgical_instrument_presence_dispara_moderate_em_streak_consecutivo():
    events = [_event(i, instrument=True) for i in range(SURGICAL_INSTRUMENT_PRESENCE_THRESHOLD)]
    trigger = rule_surgical_instrument_presence(events)
    assert trigger is not None
    # Deteccao de instrumento e NORMAL em cirurgia: dispara moderate (documentacao),
    # nunca critical sozinho. Critical vem de outros pilares (vocal/textual/facial).
    assert trigger.level == "moderate"
    assert trigger.evidence["longest_consecutive_streak"] >= SURGICAL_INSTRUMENT_PRESENCE_THRESHOLD


@pytest.mark.smoke
def test_rule_surgical_instrument_presence_dispara_moderate_em_presenca_esparsa():
    # 1 frame com instrumento em 5 = 20% (>= 10%) sem streak
    events = [
        _event(0, instrument=True),
        _event(1),
        _event(2),
        _event(3),
        _event(4),
    ]
    trigger = rule_surgical_instrument_presence(events)
    assert trigger is not None
    assert trigger.level == "moderate"


@pytest.mark.smoke
def test_rule_surgical_instrument_presence_ignora_confianca_baixa():
    events = [_event(i, instrument=True, instrument_conf=0.1) for i in range(5)]
    assert rule_surgical_instrument_presence(events) is None


@pytest.mark.smoke
def test_rule_surgical_instrument_presence_retorna_none_sem_eventos():
    assert rule_surgical_instrument_presence([]) is None


@pytest.mark.smoke
def test_rule_facial_distress_dispara_critical_em_alta_proporcao():
    events = [_event(i, facial_label="fear", facial_conf=0.9) for i in range(8)]
    events += [_event(8, facial_label="happy", facial_conf=0.9)]
    trigger = rule_facial_distress(events)
    assert trigger is not None
    assert trigger.level == "critical"


@pytest.mark.smoke
def test_rule_facial_distress_dispara_moderate_em_proporcao_intermediaria():
    events = [
        _event(0, facial_label="sad", facial_conf=0.7),
        _event(1, facial_label="happy", facial_conf=0.9),
        _event(2, facial_label="happy", facial_conf=0.9),
    ]
    trigger = rule_facial_distress(events)
    assert trigger is not None
    assert trigger.level == "moderate"


@pytest.mark.smoke
def test_rule_facial_distress_ignora_emocoes_neutras():
    events = [_event(i, facial_label="happy", facial_conf=0.95) for i in range(5)]
    assert rule_facial_distress(events) is None


# --------------------------------------------------------------------------
# Regras de audio
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_rule_vocal_distress_dispara_em_emocao_negativa_confiante():
    audio = _audio(emotion=AudioEmotionScore(label="angry", confidence=0.8, scores={"angry": 0.8}))
    trigger = rule_vocal_distress(audio)
    assert trigger is not None
    assert trigger.level == "moderate"


@pytest.mark.smoke
def test_rule_vocal_distress_nao_dispara_para_emocao_neutra():
    audio = _audio(emotion=AudioEmotionScore(label="happy", confidence=0.9, scores={"happy": 0.9}))
    assert rule_vocal_distress(audio) is None


@pytest.mark.smoke
def test_rule_vocal_strain_dispara_para_jitter_alto():
    feats = AcousticFeatures(
        duration_s=3.0,
        pitch_mean_hz=200.0,
        pitch_std_hz=10.0,
        energy_rms=0.3,
        zero_crossing_rate=0.1,
        jitter=0.08,
        shimmer=0.02,
        mfcc_means=[0.0] * 13,
    )
    trigger = rule_vocal_strain(_audio(features=feats))
    assert trigger is not None
    assert trigger.level == "moderate"
    assert trigger.evidence["jitter"] == pytest.approx(0.08)


@pytest.mark.smoke
def test_rule_vocal_strain_nao_dispara_para_features_normais():
    feats = AcousticFeatures(
        duration_s=3.0,
        pitch_mean_hz=200.0,
        pitch_std_hz=10.0,
        energy_rms=0.3,
        zero_crossing_rate=0.1,
        jitter=0.005,
        shimmer=0.01,
        mfcc_means=[0.0] * 13,
    )
    assert rule_vocal_strain(_audio(features=feats)) is None


@pytest.mark.smoke
def test_rule_low_vocal_energy_dispara_para_rms_baixo():
    feats = AcousticFeatures(
        duration_s=3.0,
        pitch_mean_hz=200.0,
        pitch_std_hz=10.0,
        energy_rms=0.005,
        zero_crossing_rate=0.1,
        jitter=0.01,
        shimmer=0.02,
        mfcc_means=[0.0] * 13,
    )
    trigger = rule_low_vocal_energy(_audio(features=feats))
    assert trigger is not None
    assert trigger.level == "moderate"


# --------------------------------------------------------------------------
# Regras de texto
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_rule_negative_sentiment_dispara_para_negative_confiante():
    audio = _audio(
        sentiment=SentimentResult(
            label="negative",
            confidence=0.85,
            scores={"negative": 0.85, "neutral": 0.1, "positive": 0.05},
        )
    )
    trigger = rule_negative_sentiment(audio)
    assert trigger is not None
    assert trigger.level == "moderate"


@pytest.mark.smoke
def test_rule_negative_sentiment_ignora_neutro():
    audio = _audio(sentiment=SentimentResult(label="neutral", confidence=0.95))
    assert rule_negative_sentiment(audio) is None


@pytest.mark.smoke
def test_rule_critical_terms_dispara_em_transcricao_com_acento():
    audio = _audio(transcription="Estou com convulsão e dor muito forte.")
    trigger = rule_critical_terms(audio)
    assert trigger is not None
    assert trigger.level == "critical"
    matched = trigger.evidence["matched_terms"]
    assert "convulsao" in matched
    assert "dor muito forte" in matched


@pytest.mark.smoke
def test_rule_critical_terms_usa_key_phrases_quando_transcricao_vazia():
    audio = _audio(key_phrases=["hemorragia recente"])
    trigger = rule_critical_terms(audio)
    assert trigger is not None
    assert trigger.evidence["matched_terms"] == ["hemorragia"]


@pytest.mark.smoke
def test_rule_critical_terms_nao_dispara_quando_nao_ha_termo():
    audio = _audio(transcription="Tudo bem, sem queixas hoje.")
    assert rule_critical_terms(audio) is None


# --------------------------------------------------------------------------
# Agregadores
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_evaluate_rules_agrega_triggers_de_video_e_audio():
    events = [_event(i, instrument=True) for i in range(SURGICAL_INSTRUMENT_PRESENCE_THRESHOLD)]
    audio = _audio(
        emotion=AudioEmotionScore(label="angry", confidence=0.8),
        transcription="Estou com sangramento.",
    )
    triggers = evaluate_rules(events, audio)
    rule_ids = {t.rule_id for t in triggers}
    assert "video.surgical_instrument_presence" in rule_ids
    assert "audio.vocal_distress" in rule_ids
    assert "text.critical_terms" in rule_ids


@pytest.mark.smoke
def test_evaluate_rules_retorna_lista_vazia_para_caso_normal():
    events = [_event(i, facial_label="happy", facial_conf=0.9) for i in range(3)]
    audio = _audio(
        transcription="Tudo bem.",
        emotion=AudioEmotionScore(label="happy", confidence=0.9),
    )
    assert evaluate_rules(events, audio) == []


@pytest.mark.smoke
def test_evaluate_rules_aceita_apenas_audio():
    audio = _audio(transcription="Senti um desmaio.")
    triggers = evaluate_rules(None, audio)
    assert any(t.rule_id == "text.critical_terms" for t in triggers)


@pytest.mark.smoke
def test_derive_risk_level_prioriza_o_mais_grave():
    triggers = [
        Trigger(rule_id="a", level="moderate", message="m", source="audio"),
        Trigger(rule_id="b", level="critical", message="c", source="text"),
        Trigger(rule_id="c", level="moderate", message="m2", source="video"),
    ]
    assert derive_risk_level(triggers) == "critical"


@pytest.mark.smoke
def test_derive_risk_level_retorna_normal_sem_triggers():
    assert derive_risk_level([]) == "normal"


@pytest.mark.smoke
def test_recommend_actions_inclui_recomendacao_padrao_quando_vazio():
    actions = recommend_actions([])
    assert actions and "Manter monitoramento padrao" in actions[0]


@pytest.mark.smoke
def test_recommend_actions_deduplica_acoes():
    triggers = [
        Trigger(
            rule_id="video.surgical_instrument_presence",
            level="moderate",
            message="x",
            source="video",
        ),
        Trigger(
            rule_id="video.surgical_instrument_sporadic",
            level="moderate",
            message="y",
            source="video",
        ),
    ]
    actions = recommend_actions(triggers)
    # Ambas as regras adicionam "Documentar..." mas a deduplicacao mantem uma so.
    assert actions.count("Documentar uso de instrumental cirurgico no prontuario.") == 1


@pytest.mark.smoke
def test_build_anomaly_result_retorna_modelo_completo():
    audio = _audio(transcription="Apresentou hemorragia.")
    result = build_anomaly_result(None, audio)
    assert isinstance(result, AnomalyResult)
    assert result.level == "critical"
    assert result.triggers
    assert result.recommended_actions


@pytest.mark.smoke
def test_build_anomaly_result_para_caso_sem_triggers_retorna_normal():
    result = build_anomaly_result(None, _audio())
    assert result.level == "normal"
    assert result.triggers == []
    assert result.explanation
    assert result.recommended_actions


# --------------------------------------------------------------------------
# Detector estatistico (Isolation Forest)
# --------------------------------------------------------------------------


def _baseline_features() -> AcousticFeatures:
    return AcousticFeatures(
        duration_s=3.0,
        pitch_mean_hz=180.0,
        pitch_std_hz=15.0,
        energy_rms=0.1,
        zero_crossing_rate=0.05,
        jitter=0.01,
        shimmer=0.02,
        mfcc_means=[0.0] * 13,
    )


@pytest.mark.smoke
def test_extract_feature_vector_retorna_todas_as_chaves():
    audio = _audio(features=_baseline_features())
    features = extract_feature_vector(None, audio)
    assert set(features.keys()) == set(FEATURE_NAMES)


@pytest.mark.smoke
def test_extract_feature_vector_calcula_surgical_instrument_ratio():
    events = [_event(0, instrument=True), _event(1)]
    features = extract_feature_vector(events, None)
    assert features["video.surgical_instrument_frame_ratio"] == pytest.approx(0.5)
    assert features["video.frame_count"] == 2.0


@pytest.mark.smoke
def test_features_to_array_preserva_ordem_canonica():
    features = {name: float(idx) for idx, name in enumerate(FEATURE_NAMES)}
    arr = features_to_array(features)
    assert arr == [float(i) for i in range(len(FEATURE_NAMES))]


@pytest.mark.smoke
def test_statistical_detector_retorna_none_antes_de_fit():
    detector = StatisticalAnomalyDetector()
    features = extract_feature_vector(None, _audio(features=_baseline_features()))
    assert detector.score(features) is None
    assert detector.is_anomalous(features) is False


@pytest.mark.smoke
def test_statistical_detector_pontua_caso_anomalo_acima_de_baseline():
    detector = StatisticalAnomalyDetector(contamination=0.2, random_state=0)
    # 50 amostras "normais" em torno do baseline; pequena variabilidade gaussiana.
    rng_seed = 17
    baseline = extract_feature_vector(None, _audio(features=_baseline_features()))
    samples: list[dict[str, float]] = []
    for i in range(50):
        sample = dict(baseline)
        sample["audio.pitch_mean_hz"] += ((i * rng_seed) % 7) * 0.3
        sample["audio.pitch_std_hz"] += ((i * rng_seed) % 5) * 0.2
        sample["audio.energy_rms"] += ((i * rng_seed) % 3) * 0.002
        sample["audio.jitter"] += ((i * rng_seed) % 4) * 0.001
        samples.append(sample)
    detector.fit(samples)
    assert detector.is_fitted

    anomaly = dict(baseline)
    anomaly["audio.jitter"] = 0.5
    anomaly["audio.shimmer"] = 0.9
    anomaly["audio.pitch_mean_hz"] = 9999.0
    anomaly["audio.energy_rms"] = 5.0
    score_normal = detector.score(baseline)
    score_anomaly = detector.score(anomaly)
    assert score_normal is not None
    assert score_anomaly is not None
    # Caso extremo deve receber score maior; ambos em [0, 1].
    assert 0.0 <= score_normal <= 1.0
    assert 0.0 <= score_anomaly <= 1.0
    assert score_anomaly > score_normal


@pytest.mark.smoke
def test_statistical_detector_fit_rejeita_amostras_vazias():
    detector = StatisticalAnomalyDetector()
    with pytest.raises(ValueError):
        detector.fit([])


# --------------------------------------------------------------------------
# Classificador final
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_classifier_sem_detector_estatistico_usa_so_regras():
    classifier = AnomalyClassifier()
    audio = _audio(transcription="Apresentou hemorragia.")
    result = classifier.classify(None, audio)
    assert isinstance(result, AnomalyResult)
    assert result.level == "critical"
    assert any(t.rule_id == "text.critical_terms" for t in result.triggers)
    assert all(t.source != "fusion" for t in result.triggers)


@pytest.mark.smoke
def test_classifier_caso_normal_retorna_normal_sem_triggers():
    classifier = AnomalyClassifier()
    result = classifier.classify(None, _audio())
    assert result.level == "normal"
    assert result.triggers == []


@pytest.mark.smoke
def test_classifier_inclui_trigger_estatistico_quando_score_alto():
    detector = StatisticalAnomalyDetector(contamination=0.2, random_state=0)
    baseline = extract_feature_vector(None, _audio(features=_baseline_features()))
    samples = [dict(baseline) for _ in range(50)]
    for i, s in enumerate(samples):
        s["audio.pitch_mean_hz"] += (i % 5) * 0.5
    detector.fit(samples)
    # Forcamos score alto via threshold baixo (0.0) para verificar caminho.
    classifier = AnomalyClassifier(statistical_detector=detector, statistical_threshold=0.0)
    result = classifier.classify(None, _audio(features=_baseline_features()))
    assert any(t.rule_id == "statistical.outlier" for t in result.triggers)


@pytest.mark.smoke
def test_classifier_nao_inclui_estatistico_quando_detector_nao_treinado():
    detector = StatisticalAnomalyDetector()
    classifier = AnomalyClassifier(statistical_detector=detector)
    result = classifier.classify(None, _audio(features=_baseline_features()))
    assert all(t.rule_id != "statistical.outlier" for t in result.triggers)


@pytest.mark.smoke
def test_classifier_preserva_critical_acima_de_estatistico_moderate():
    detector = StatisticalAnomalyDetector(contamination=0.2, random_state=0)
    baseline = extract_feature_vector(None, _audio(features=_baseline_features()))
    detector.fit([dict(baseline) for _ in range(30)])
    classifier = AnomalyClassifier(statistical_detector=detector, statistical_threshold=0.0)
    # Regra critica garantida via termo "hemorragia"
    audio = _audio(transcription="Apresentou hemorragia.", features=_baseline_features())
    result = classifier.classify(None, audio)
    assert result.level == "critical"
