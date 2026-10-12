"""Regras clinicas explicitas para deteccao de anomalia.

Cada regra e uma funcao pura que recebe a saida agregada dos pipelines
de video e audio (`list[VideoEvent]` e `AudioAnalysis`) e devolve um
`Trigger` quando o criterio clinico e atingido, ou `None` caso contrario.

Regras tem prioridade sobre o modelo estatistico porque sao auditaveis
e diretamente justificaveis em contexto clinico. `AnomalyClassifier`
combina ambos preservando triggers criticos.

Thresholds vivem como constantes no topo do modulo. Valores iniciais
sao calibrados em literatura clinica leve (jitter/shimmer) e em
proporcoes empiricas (frames com distress).
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable, Iterable

from src.anomaly.types import AnomalyResult, RiskLevel, Trigger
from src.audio.types import AudioAnalysis
from src.video.types import VideoEvent

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes de tuning
# --------------------------------------------------------------------------

# Ordem numerica usada para escolher o trigger mais grave em `derive_risk_level`.
RISK_LEVEL_ORDER: dict[RiskLevel, int] = {"normal": 0, "moderate": 1, "critical": 2}

# Instrumentos cirurgicos laparoscopicos (detector custom treinado em CholecSeg8k).
# Deteccao de instrumento e NORMAL em cirurgia; regra so dispara moderate
# (documentacao de procedimento), critical vem de outros pilares (vocal/textual/facial).
SURGICAL_INSTRUMENT_CLASS_NAMES: frozenset[str] = frozenset(
    {"grasper", "l_hook_electrocautery", "hook"}
)
SURGICAL_INSTRUMENT_MIN_CONFIDENCE: float = 0.5
SURGICAL_INSTRUMENT_PRESENCE_THRESHOLD: int = 3
SURGICAL_INSTRUMENT_MODERATE_FRAME_RATIO: float = 0.1

# Sangramento detectado em video cirurgico (classe blood do YOLO custom).
# Sangue em cirurgia laparoscopica = sinal de complicacao real; regra dispara
# critical ao detectar presenca persistente em frames consecutivos.
BLEEDING_CLASS_NAMES: frozenset[str] = frozenset({"blood", "bleeding"})
BLEEDING_MIN_CONFIDENCE: float = 0.4
BLEEDING_PRESENCE_THRESHOLD: int = 2
BLEEDING_CRITICAL_FRAME_RATIO: float = 0.05

# Distress facial agregado pelo pipeline de video (FER).
FACIAL_DISTRESS_LABELS: frozenset[str] = frozenset({"fear", "sad", "angry", "disgust"})
FACIAL_DISTRESS_MIN_CONFIDENCE: float = 0.55
FACIAL_DISTRESS_RATIO_MODERATE: float = 0.3
FACIAL_DISTRESS_RATIO_CRITICAL: float = 0.6

# Emocao vocal (wav2vec2-superb-er usa labels curtos: ang, hap, neu, sad).
VOCAL_DISTRESS_LABELS: frozenset[str] = frozenset({"angry", "sad", "fear", "fearful", "ang", "fea"})
VOCAL_DISTRESS_MIN_CONFIDENCE: float = 0.55

# Features acusticas (proxies de tensao vocal segundo literatura voice quality).
JITTER_HIGH_THRESHOLD: float = 0.04
SHIMMER_HIGH_THRESHOLD: float = 0.08
ENERGY_RMS_LOW_THRESHOLD: float = 0.015

# Sentimento textual (Azure Language retorna labels positive/neutral/negative).
SENTIMENT_NEGATIVE_MIN_CONFIDENCE: float = 0.7

# Termos clinicos criticos esperados na transcricao ou nas key phrases.
# Normalizados (sem acento, lower) para casar com `_normalize`.
CRITICAL_TERMS: frozenset[str] = frozenset(
    {
        "sangramento",
        "hemorragia",
        "desmaio",
        "desmaiei",
        "perda de consciencia",
        "convulsao",
        "convulsoes",
        "dor insuportavel",
        "dor muito forte",
        "nao consigo respirar",
        "falta de ar severa",
        "tontura intensa",
        "visao turva",
        "pressao alta",
    }
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Remove acentos e baixa caixa para comparar termos em PT-BR.

    Whisper e key-phrase extractors costumam variar em diacriticos. Comparar
    versoes normalizadas evita falhar por causa de "convulsao" vs "convulsão".
    """
    normalized = unicodedata.normalize("NFD", text)
    no_accents = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return no_accents.lower()


def _max_level(triggers: Iterable[Trigger]) -> RiskLevel:
    """Retorna o nivel mais grave entre os triggers; `normal` se lista vazia."""
    highest: RiskLevel = "normal"
    highest_score = RISK_LEVEL_ORDER[highest]
    for trigger in triggers:
        score = RISK_LEVEL_ORDER[trigger.level]
        if score > highest_score:
            highest = trigger.level
            highest_score = score
    return highest


def _longest_consecutive_streak(flags: list[bool]) -> int:
    """Maior sequencia de `True` consecutivos em uma lista de flags."""
    longest = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# --------------------------------------------------------------------------
# Regras de video
# --------------------------------------------------------------------------


def rule_surgical_instrument_presence(events: list[VideoEvent]) -> Trigger | None:
    """Documenta presenca persistente de instrumental cirurgico no video.

    Diferente das regras de anomalia visual (sangramento, distress facial),
    a deteccao de instrumento e ESPERADA em cirurgia. Por isso a regra so
    dispara `moderate` (registro/documentacao de procedimento). Alertas
    `critical` vem de outras regras (vocal_distress, critical_terms,
    facial_distress) cruzadas com o contexto cirurgico.
    """
    if not events:
        return None

    instrument_flags: list[bool] = []
    confident_count = 0
    for event in events:
        has_instrument = any(
            det.class_name.lower() in SURGICAL_INSTRUMENT_CLASS_NAMES
            and det.confidence >= SURGICAL_INSTRUMENT_MIN_CONFIDENCE
            for det in event.detections
        )
        instrument_flags.append(has_instrument)
        if has_instrument:
            confident_count += 1

    if confident_count == 0:
        return None

    longest_streak = _longest_consecutive_streak(instrument_flags)
    ratio = confident_count / len(events)

    evidence = {
        "frames_with_instrument": confident_count,
        "total_frames": len(events),
        "longest_consecutive_streak": longest_streak,
        "ratio": round(ratio, 3),
    }

    if longest_streak >= SURGICAL_INSTRUMENT_PRESENCE_THRESHOLD:
        return Trigger(
            rule_id="video.surgical_instrument_presence",
            level="moderate",
            message=(
                f"Instrumental cirurgico em uso em {longest_streak} frames consecutivos"
                f" (>= {SURGICAL_INSTRUMENT_PRESENCE_THRESHOLD})."
            ),
            source="video",
            evidence=evidence,
        )
    if ratio >= SURGICAL_INSTRUMENT_MODERATE_FRAME_RATIO:
        return Trigger(
            rule_id="video.surgical_instrument_sporadic",
            level="moderate",
            message=(
                f"Instrumental cirurgico em uso em {confident_count}/{len(events)} frames"
                f" ({ratio:.1%} do video)."
            ),
            source="video",
            evidence=evidence,
        )
    return None


def rule_bleeding_detected(events: list[VideoEvent]) -> Trigger | None:
    """Alerta sangramento detectado em video cirurgico.

    Diferente da regra de instrumento (descritiva, `moderate`), sangramento
    persistente em cirurgia laparoscopica e um sinal de complicacao real e
    dispara `critical`. Threshold de confianca menor que o de instrumento
    porque sangue tem forma irregular e e mais dificil de classificar com
    alta confianca pelo YOLO.
    """
    if not events:
        return None

    bleeding_flags: list[bool] = []
    confident_count = 0
    for event in events:
        has_bleeding = any(
            det.class_name.lower() in BLEEDING_CLASS_NAMES
            and det.confidence >= BLEEDING_MIN_CONFIDENCE
            for det in event.detections
        )
        bleeding_flags.append(has_bleeding)
        if has_bleeding:
            confident_count += 1

    if confident_count == 0:
        return None

    longest_streak = _longest_consecutive_streak(bleeding_flags)
    ratio = confident_count / len(events)

    evidence = {
        "frames_with_bleeding": confident_count,
        "total_frames": len(events),
        "longest_consecutive_streak": longest_streak,
        "ratio": round(ratio, 3),
    }

    if (
        longest_streak >= BLEEDING_PRESENCE_THRESHOLD
        or ratio >= BLEEDING_CRITICAL_FRAME_RATIO
    ):
        return Trigger(
            rule_id="video.bleeding_detected",
            level="critical",
            message=(
                f"Sangramento detectado em {confident_count}/{len(events)} frames"
                f" ({ratio:.1%}), maior streak consecutivo: {longest_streak}."
            ),
            source="video",
            evidence=evidence,
        )
    return None


def rule_facial_distress(events: list[VideoEvent]) -> Trigger | None:
    """Sinaliza distress facial recorrente.

    Conta frames cuja emocao predominante e negativa
    (`FACIAL_DISTRESS_LABELS`) com confianca acima de
    `FACIAL_DISTRESS_MIN_CONFIDENCE`, e dispara `moderate` ou `critical`
    conforme o ratio.
    """
    distress_count = 0
    valid_events = 0
    labels_observed: dict[str, int] = {}

    for event in events:
        emotion = event.facial_emotion
        if emotion is None:
            continue
        valid_events += 1
        label = emotion.label.lower()
        if label in FACIAL_DISTRESS_LABELS and emotion.confidence >= FACIAL_DISTRESS_MIN_CONFIDENCE:
            distress_count += 1
            labels_observed[label] = labels_observed.get(label, 0) + 1

    if valid_events == 0 or distress_count == 0:
        return None

    ratio = distress_count / valid_events
    evidence = {
        "distress_frames": distress_count,
        "valid_frames": valid_events,
        "ratio": round(ratio, 3),
        "labels": labels_observed,
    }

    if ratio >= FACIAL_DISTRESS_RATIO_CRITICAL:
        return Trigger(
            rule_id="video.facial_distress",
            level="critical",
            message=(
                f"Distress facial em {ratio:.1%} dos frames com face"
                f" (>= {FACIAL_DISTRESS_RATIO_CRITICAL:.0%})."
            ),
            source="video",
            evidence=evidence,
        )
    if ratio >= FACIAL_DISTRESS_RATIO_MODERATE:
        return Trigger(
            rule_id="video.facial_distress",
            level="moderate",
            message=(
                f"Distress facial em {ratio:.1%} dos frames com face"
                f" (>= {FACIAL_DISTRESS_RATIO_MODERATE:.0%})."
            ),
            source="video",
            evidence=evidence,
        )
    return None


# --------------------------------------------------------------------------
# Regras de audio
# --------------------------------------------------------------------------


def rule_vocal_distress(audio: AudioAnalysis) -> Trigger | None:
    """Sinaliza emocao vocal negativa com alta confianca."""
    emotion = audio.emotion
    if emotion is None:
        return None
    label = emotion.label.lower()
    if label in VOCAL_DISTRESS_LABELS and emotion.confidence >= VOCAL_DISTRESS_MIN_CONFIDENCE:
        return Trigger(
            rule_id="audio.vocal_distress",
            level="moderate",
            message=(
                f"Emocao vocal '{emotion.label_pt}' detectada com confianca "
                f"{emotion.confidence:.2f}."
            ),
            source="audio",
            evidence={
                "label": emotion.label,
                "label_pt": emotion.label_pt,
                "confidence": round(emotion.confidence, 3),
                "scores": emotion.scores,
            },
        )
    return None


def rule_vocal_strain(audio: AudioAnalysis) -> Trigger | None:
    """Sinaliza tensao vocal por jitter e/ou shimmer elevados."""
    features = audio.acoustic_features
    if features is None:
        return None
    high_jitter = features.jitter >= JITTER_HIGH_THRESHOLD
    high_shimmer = features.shimmer >= SHIMMER_HIGH_THRESHOLD
    if not (high_jitter or high_shimmer):
        return None
    return Trigger(
        rule_id="audio.vocal_strain",
        level="moderate",
        message=(
            f"Tensao vocal acima do esperado: jitter={features.jitter:.3f},"
            f" shimmer={features.shimmer:.3f}."
        ),
        source="audio",
        evidence={
            "jitter": round(features.jitter, 4),
            "shimmer": round(features.shimmer, 4),
            "jitter_threshold": JITTER_HIGH_THRESHOLD,
            "shimmer_threshold": SHIMMER_HIGH_THRESHOLD,
        },
    )


def rule_low_vocal_energy(audio: AudioAnalysis) -> Trigger | None:
    """Sinaliza energia vocal muito baixa (possivel exaustao ou apatia)."""
    features = audio.acoustic_features
    if features is None:
        return None
    if features.energy_rms >= ENERGY_RMS_LOW_THRESHOLD:
        return None
    return Trigger(
        rule_id="audio.low_energy",
        level="moderate",
        message=(
            f"Energia vocal baixa (RMS={features.energy_rms:.4f} < {ENERGY_RMS_LOW_THRESHOLD})."
        ),
        source="audio",
        evidence={
            "energy_rms": round(features.energy_rms, 5),
            "threshold": ENERGY_RMS_LOW_THRESHOLD,
        },
    )


# --------------------------------------------------------------------------
# Regras de texto
# --------------------------------------------------------------------------


def rule_negative_sentiment(audio: AudioAnalysis) -> Trigger | None:
    """Sinaliza sentimento textual negativo da transcricao."""
    sentiment = audio.sentiment
    if sentiment is None:
        return None
    if (
        sentiment.label.lower() == "negative"
        and sentiment.confidence >= SENTIMENT_NEGATIVE_MIN_CONFIDENCE
    ):
        return Trigger(
            rule_id="text.negative_sentiment",
            level="moderate",
            message=(f"Sentimento textual negativo (confianca {sentiment.confidence:.2f})."),
            source="text",
            evidence={
                "confidence": round(sentiment.confidence, 3),
                "scores": sentiment.scores,
            },
        )
    return None


def rule_critical_terms(audio: AudioAnalysis) -> Trigger | None:
    """Sinaliza presenca de termos clinicos criticos na transcricao.

    Olha tanto a transcricao completa quanto as `key_phrases` extraidas
    pelo Azure Language. Considera um termo presente quando aparece como
    substring na versao normalizada (sem acento, lower).
    """
    haystacks: list[str] = []
    if audio.transcription:
        haystacks.append(_normalize(audio.transcription))
    if audio.key_phrases:
        haystacks.append(_normalize(" ".join(audio.key_phrases)))
    if not haystacks:
        return None

    matched = sorted({term for term in CRITICAL_TERMS if any(term in hay for hay in haystacks)})
    if not matched:
        return None

    return Trigger(
        rule_id="text.critical_terms",
        level="critical",
        message=f"Termos clinicos criticos detectados: {', '.join(matched)}.",
        source="text",
        evidence={"matched_terms": matched},
    )


# --------------------------------------------------------------------------
# Agregadores
# --------------------------------------------------------------------------

VideoRule = Callable[[list[VideoEvent]], Trigger | None]
AudioRule = Callable[[AudioAnalysis], Trigger | None]

VIDEO_RULES: tuple[VideoRule, ...] = (
    rule_surgical_instrument_presence,
    rule_bleeding_detected,
    rule_facial_distress,
)

AUDIO_RULES: tuple[AudioRule, ...] = (
    rule_vocal_distress,
    rule_vocal_strain,
    rule_low_vocal_energy,
    rule_negative_sentiment,
    rule_critical_terms,
)


def evaluate_rules(
    video_events: list[VideoEvent] | None = None,
    audio_analysis: AudioAnalysis | None = None,
) -> list[Trigger]:
    """Aplica todas as regras nas entradas disponiveis e devolve os triggers.

    Args:
        video_events: lista de `VideoEvent` produzida pelo pipeline de video.
            Pode ser `None` ou vazia quando so houver audio.
        audio_analysis: `AudioAnalysis` produzida pelo pipeline de audio.
            Pode ser `None` quando so houver video.

    Returns:
        Lista de `Trigger` na ordem em que as regras foram aplicadas.
        Lista vazia significa que nenhuma regra clinica disparou.
    """
    triggers: list[Trigger] = []

    if video_events:
        for rule in VIDEO_RULES:
            trigger = rule(video_events)
            if trigger is not None:
                triggers.append(trigger)

    if audio_analysis is not None:
        for rule in AUDIO_RULES:
            trigger = rule(audio_analysis)
            if trigger is not None:
                triggers.append(trigger)

    logger.info(
        "Regras avaliadas: %d trigger(s) disparado(s) (%s).",
        len(triggers),
        ", ".join(t.rule_id for t in triggers) or "nenhum",
    )
    return triggers


def derive_risk_level(triggers: list[Trigger]) -> RiskLevel:
    """Deduz o nivel de risco final a partir do trigger mais grave."""
    return _max_level(triggers)


def recommend_actions(triggers: list[Trigger]) -> list[str]:
    """Sugere acoes clinicas baseadas nos triggers disparados.

    Lista preliminar de orientacoes em PT-BR. O relatorio final usa LLM
    para texto naturalizado; este e o fallback deterministico quando o
    LLM nao esta disponivel.
    """
    actions: list[str] = []
    seen: set[str] = set()

    def _add(action: str) -> None:
        if action not in seen:
            actions.append(action)
            seen.add(action)

    for trigger in triggers:
        if trigger.rule_id.startswith("video.surgical_instrument"):
            _add("Documentar uso de instrumental cirurgico no prontuario.")
            _add("Correlacionar com fase cirurgica e contexto clinico do procedimento.")
        elif trigger.rule_id == "video.bleeding_detected":
            _add("Acionar protocolo de hemorragia: avaliar fonte e magnitude.")
            _add("Verificar parametros hemodinamicos e necessidade de transfusao.")
            _add("Reforcar equipe cirurgica caso intraoperatorio.")
        elif trigger.rule_id == "video.facial_distress":
            _add("Avaliar quadro emocional e oferecer suporte psicologico.")
        elif trigger.rule_id == "audio.vocal_distress":
            _add("Conduzir entrevista clinica focada em estado emocional.")
        elif trigger.rule_id == "audio.vocal_strain":
            _add("Investigar fadiga, ansiedade ou dor recente.")
        elif trigger.rule_id == "audio.low_energy":
            _add("Avaliar exaustao, anemia ou sintomas depressivos.")
        elif trigger.rule_id == "text.negative_sentiment":
            _add("Aprofundar anamnese sobre queixas relatadas.")
        elif trigger.rule_id == "text.critical_terms":
            _add("Acionar protocolo de urgencia conforme termos relatados.")
            _add("Encaminhar para avaliacao medica imediata.")

    if not actions:
        _add("Manter monitoramento padrao; nenhum sinal de alerta detectado.")
    return actions


def build_anomaly_result(
    video_events: list[VideoEvent] | None = None,
    audio_analysis: AudioAnalysis | None = None,
) -> AnomalyResult:
    """Atalho que aplica regras, deduz nivel e monta `AnomalyResult`.

    `AnomalyClassifier` cobre o caminho completo (regras + Isolation
    Forest); esta funcao serve como base de regras quando o detector
    estatistico nao e desejado.
    """
    triggers = evaluate_rules(video_events, audio_analysis)
    level = derive_risk_level(triggers)
    actions = recommend_actions(triggers)
    if triggers:
        explanation = "; ".join(t.message for t in triggers)
    else:
        explanation = "Nenhuma regra clinica disparada."
    return AnomalyResult(
        level=level,
        triggers=triggers,
        explanation=explanation,
        recommended_actions=actions,
    )
