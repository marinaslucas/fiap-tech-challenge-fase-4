"""Testes de integracao do orquestrador `process_case`.

Mocks substituem pipelines pesados (video/audio) e clients externos
(LLM, RAG) para que cada teste rode em <1s mas exercite o fluxo
completo: pipeline -> classifier -> RAG -> report -> alert -> audit.

Cobertura: 3 cenarios principais (normal, moderado, critico) + casos
de borda (so audio, so video, sem nada).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.alert import Alert
from src.anomaly import AnomalyClassifier
from src.audio.types import AcousticFeatures, AudioAnalysis, EmotionScore, SentimentResult
from src.audit import AuditLogger
from src.orchestrator import CaseInput, Orchestrator
from src.rag.types import Chunk, RetrievalResult
from src.video.types import BoundingBox, Detection, VideoEvent

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _audio(
    *,
    transcription: str = "",
    emotion: EmotionScore | None = None,
    sentiment: SentimentResult | None = None,
    features: AcousticFeatures | None = None,
) -> AudioAnalysis:
    return AudioAnalysis(
        transcription=transcription,
        segments=[],
        acoustic_features=features,
        emotion=emotion,
        sentiment=sentiment,
        key_phrases=[],
    )


def _video_events(*, instrument: int = 0, total: int = 5) -> list[VideoEvent]:
    events: list[VideoEvent] = []
    for i in range(total):
        detections = []
        if i < instrument:
            detections.append(
                Detection(
                    class_id=0,
                    class_name="grasper",
                    confidence=0.9,
                    bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                )
            )
        events.append(
            VideoEvent(
                frame_index=i,
                timestamp_ms=i * 1000,
                detections=detections,
                facial_emotion=None,
            )
        )
    return events


def _mock_video_pipeline(events: list[VideoEvent]) -> MagicMock:
    from src.video.scene_classifier import SceneType
    pipeline = MagicMock()
    pipeline.process.return_value = events
    pipeline.last_scene_type = SceneType.UNKNOWN
    return pipeline


def _mock_audio_pipeline(analysis: AudioAnalysis) -> MagicMock:
    pipeline = MagicMock()
    pipeline.process.return_value = analysis
    return pipeline


def _mock_retriever(chunks: list[Chunk]) -> MagicMock:
    retriever = MagicMock()
    result = RetrievalResult(chunks=chunks, scores=[0.9] * len(chunks))
    retriever.search.return_value = result
    retriever.multi_search.return_value = result
    return retriever


def _llm_unconfigured() -> MagicMock:
    """Cliente que reporta nao configurado (forca fallback do report)."""
    client = MagicMock()
    client.is_configured = False
    return client


def _orchestrator(
    *,
    tmp_path: Path,
    video_events: list[VideoEvent] | None,
    audio_analysis: AudioAnalysis | None,
    rag_chunks: list[Chunk] | None = None,
) -> Orchestrator:
    chunks = rag_chunks or [
        Chunk(
            text="Diretriz teste: investigar sangramento via exame fisico.",
            source="manual_ms.pdf",
            section="prenatal",
            page=1,
            chunk_id="manual_ms.pdf::1::0",
        )
    ]
    return Orchestrator(
        video_pipeline=_mock_video_pipeline(video_events or []),
        audio_pipeline=_mock_audio_pipeline(audio_analysis or _audio()),
        classifier=AnomalyClassifier(),
        retriever=_mock_retriever(chunks),
        llm_client=_llm_unconfigured(),
        auditor=AuditLogger(db_path=tmp_path / "audit.sqlite"),
    )


# --------------------------------------------------------------------------
# Cenarios principais
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_cenario_normal_sem_triggers(tmp_path):
    orch = _orchestrator(
        tmp_path=tmp_path,
        video_events=_video_events(instrument=0, total=3),
        audio_analysis=_audio(transcription="Tudo bem."),
    )
    case = CaseInput(
        video_path=Path("/fake/video.mp4"),
        audio_path=Path("/fake/audio.wav"),
        patient_metadata={"id": "P1"},
    )
    out = orch.process_case(case)

    assert out.anomaly.level == "normal"
    assert out.anomaly.triggers == []
    assert out.alert is None
    assert out.audit_id is not None
    assert out.report_markdown.startswith("# Relatorio Clinico Automatico")
    assert "case-" in out.case_id


@pytest.mark.integration
def test_cenario_moderado_audio_negativo(tmp_path):
    orch = _orchestrator(
        tmp_path=tmp_path,
        video_events=_video_events(instrument=0, total=3),
        audio_analysis=_audio(
            sentiment=SentimentResult(label="negative", confidence=0.9),
            emotion=EmotionScore(label="sad", confidence=0.8),
        ),
    )
    case = CaseInput(audio_path=Path("/fake/audio.wav"), patient_metadata={"id": "P2"})
    out = orch.process_case(case)

    assert out.anomaly.level == "moderate"
    assert any(t.rule_id == "text.negative_sentiment" for t in out.anomaly.triggers)
    assert isinstance(out.alert, Alert)
    assert out.alert.level == "moderate"
    assert out.audit_id is not None


@pytest.mark.integration
def test_cenario_critico_instrumento_e_termo(tmp_path):
    orch = _orchestrator(
        tmp_path=tmp_path,
        video_events=_video_events(instrument=4, total=6),
        audio_analysis=_audio(transcription="Apresentou hemorragia intensa."),
    )
    case = CaseInput(
        video_path=Path("/fake/video.mp4"),
        audio_path=Path("/fake/audio.wav"),
        patient_metadata={"id": "P3"},
    )
    out = orch.process_case(case)

    assert out.anomaly.level == "critical"
    rule_ids = {t.rule_id for t in out.anomaly.triggers}
    assert "video.surgical_instrument_presence" in rule_ids
    assert "text.critical_terms" in rule_ids
    assert out.alert is not None
    assert out.alert.level == "critical"

    # Audit persiste o caso
    record = orch.auditor.get_case(out.audit_id)
    assert record is not None
    assert record["risk_level"] == "critical"
    assert "video" in record["modalities"]


@pytest.mark.integration
def test_caso_so_audio_funciona_sem_video(tmp_path):
    orch = _orchestrator(
        tmp_path=tmp_path,
        video_events=None,
        audio_analysis=_audio(transcription="Sangramento leve."),
    )
    out = orch.process_case(CaseInput(audio_path=Path("/fake/audio.wav")))
    assert out.video_events is None
    assert out.audio_analysis is not None
    assert out.anomaly.level == "critical"


@pytest.mark.integration
def test_caso_so_video_funciona_sem_audio(tmp_path):
    """So video + instrumento detectado: nivel `moderate` (deteccao de
    instrumento e documentacao, nao alerta critico). Critical exige sinal
    humano (vocal/textual/facial).
    """
    orch = _orchestrator(
        tmp_path=tmp_path,
        video_events=_video_events(instrument=3, total=5),
        audio_analysis=None,
    )
    out = orch.process_case(CaseInput(video_path=Path("/fake/video.mp4")))
    assert out.audio_analysis is None
    assert out.video_events is not None
    assert out.anomaly.level == "moderate"


@pytest.mark.integration
def test_caso_sem_inputs_retorna_normal(tmp_path):
    orch = _orchestrator(tmp_path=tmp_path, video_events=None, audio_analysis=None)
    out = orch.process_case(CaseInput())
    assert out.anomaly.level == "normal"
    assert out.alert is None
    assert out.report_markdown


@pytest.mark.integration
def test_relatorio_inclui_diretrizes_rag(tmp_path):
    chunks = [
        Chunk(
            text="Conduta em sangramento: avaliacao imediata.",
            source="manual_ms.pdf",
            chunk_id="manual_ms.pdf::1::0",
        ),
    ]
    orch = _orchestrator(
        tmp_path=tmp_path,
        video_events=None,
        audio_analysis=_audio(transcription="Apresentou hemorragia."),
        rag_chunks=chunks,
    )
    out = orch.process_case(CaseInput(audio_path=Path("/fake/audio.wav")))
    assert "manual_ms.pdf" in out.report_markdown
    assert out.rag_context == chunks
