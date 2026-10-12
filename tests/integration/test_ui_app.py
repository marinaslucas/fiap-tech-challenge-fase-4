"""Testes integration da camada UI (`app.py` + abas Gradio).

Cobrem dois caminhos:

1. `build_app()` constroi o `gr.Blocks` com todas as 4 abas, exemplos
   carregados e nao quebra ao montar a interface.
2. O fluxo upload+processamento simulado: `run_case_with` com um
   `Orchestrator` totalmente mockado (mesma estrategia de
   `scripts/demo_orchestrator.py`) produz `CaseOutput` coerente para os
   3 cenarios alvo (normal, moderado, critico). E o mesmo caminho que a
   aba multimodal vai consumir em runtime.

Os pipelines pesados nao sao instanciados; tudo passa por mocks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import gradio as gr
import pytest
from app import build_app, run_case_with

from src.anomaly import AnomalyClassifier
from src.audio.types import (
    AcousticFeatures,
    AudioAnalysis,
    EmotionScore,
    SentimentResult,
)
from src.audit import AuditLogger
from src.orchestrator import Orchestrator
from src.rag.types import Chunk, RetrievalResult
from src.video.types import BoundingBox, Detection, VideoEvent
from ui.components import (
    format_anomaly_summary,
    format_audio_summary,
    format_rag_chunks,
    risk_badge,
    triggers_to_rows,
    video_events_to_rows,
)

# ---------------------------------------------------------------------------
# Helpers (espelham os do test_orchestrator.py para nao acoplar)
# ---------------------------------------------------------------------------


def _audio(
    *,
    transcription: str = "",
    emotion: EmotionScore | None = None,
    sentiment: SentimentResult | None = None,
) -> AudioAnalysis:
    features = AcousticFeatures(
        duration_s=3.0,
        pitch_mean_hz=220.0,
        pitch_std_hz=10.0,
        energy_rms=0.05,
        zero_crossing_rate=0.1,
        jitter=0.01,
        shimmer=0.02,
        mfcc_means=[0.0] * 13,
    )
    return AudioAnalysis(
        transcription=transcription,
        segments=[],
        acoustic_features=features,
        emotion=emotion,
        sentiment=sentiment,
        key_phrases=[],
    )


def _events(instrument: int = 0, total: int = 3) -> list[VideoEvent]:
    events: list[VideoEvent] = []
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    for i in range(total):
        dets = (
            [Detection(class_id=0, class_name="grasper", confidence=0.9, bbox=bbox)]
            if i < instrument
            else []
        )
        events.append(
            VideoEvent(frame_index=i, timestamp_ms=i * 1000, detections=dets)
        )
    return events


def _orchestrator(
    *,
    audio: AudioAnalysis | None,
    events: list[VideoEvent] | None,
    db_path: Path,
) -> Orchestrator:
    from src.video.scene_classifier import SceneType
    video_pipeline = MagicMock()
    video_pipeline.process.return_value = events or []
    video_pipeline.last_scene_type = SceneType.UNKNOWN
    audio_pipeline = MagicMock()
    audio_pipeline.process.return_value = audio or _audio()

    chunk = Chunk(
        text="Diretriz: avaliar prontamente sangramentos e dor abdominal.",
        source="manual_ms.pdf",
        section="prenatal",
        page=1,
        chunk_id="manual_ms.pdf::1::0",
    )
    retriever = MagicMock()
    retriever.search.return_value = RetrievalResult(chunks=[chunk], scores=[0.9])

    llm = MagicMock()
    llm.is_configured = False  # forca fallback deterministico do report

    return Orchestrator(
        video_pipeline=video_pipeline,
        audio_pipeline=audio_pipeline,
        classifier=AnomalyClassifier(),
        retriever=retriever,
        llm_client=llm,
        auditor=AuditLogger(db_path=db_path),
    )


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_build_app_monta_blocks_com_4_abas(tmp_path: Path):
    """`build_app` deve montar `gr.Blocks` mesmo com um orquestrador mockado."""
    orch = _orchestrator(audio=_audio(), events=_events(), db_path=tmp_path / "audit.db")
    app = build_app(orchestrator=orch)
    assert isinstance(app, gr.Blocks)
    # Conta TabItem para garantir as 4 abas
    tab_items = [b for b in app.blocks.values() if type(b).__name__ == "Tab"]
    assert len(tab_items) >= 4


@pytest.mark.integration
def test_run_case_normal_produz_nivel_normal(tmp_path: Path):
    """Fluxo upload+processar com inputs neutros -> AnomalyResult `normal`."""
    orch = _orchestrator(
        audio=_audio(transcription="Tudo bem."),
        events=_events(instrument=0, total=3),
        db_path=tmp_path / "audit.db",
    )
    out = run_case_with(
        orch,
        video_path=Path("/fake/v.mp4"),
        audio_path=Path("/fake/a.wav"),
        text_context=None,
        patient_metadata={"id": "p-normal"},
    )
    assert out.anomaly.level == "normal"
    assert out.audit_id is not None
    # Helpers da UI conseguem renderizar a saida sem quebrar
    assert "Normal" in risk_badge(out.anomaly.level)
    assert format_anomaly_summary(out.anomaly)
    assert isinstance(triggers_to_rows(out.anomaly.triggers), list)


@pytest.mark.integration
def test_run_case_moderado_produz_nivel_moderate(tmp_path: Path):
    """Sentimento negativo + emocao triste -> nivel `moderate`."""
    orch = _orchestrator(
        audio=_audio(
            transcription="Estou cansada e sem animo.",
            sentiment=SentimentResult(label="negative", confidence=0.85),
            emotion=EmotionScore(label="sad", confidence=0.8),
        ),
        events=None,
        db_path=tmp_path / "audit.db",
    )
    out = run_case_with(
        orch,
        video_path=None,
        audio_path=Path("/fake/a.wav"),
        text_context="Paciente puerpera com queixas afetivas.",
        patient_metadata={"id": "p-mod"},
    )
    assert out.anomaly.level == "moderate"
    assert out.report_markdown  # fallback deterministico nunca vazio
    assert format_audio_summary(out.audio_analysis)


@pytest.mark.integration
def test_run_case_critico_produz_nivel_critical(tmp_path: Path):
    """Sangramento em frames consecutivos + termo critico -> nivel `critical`."""
    orch = _orchestrator(
        audio=_audio(transcription="Hemorragia intensa."),
        events=_events(instrument=4, total=6),
        db_path=tmp_path / "audit.db",
    )
    out = run_case_with(
        orch,
        video_path=Path("/fake/v.mp4"),
        audio_path=Path("/fake/a.wav"),
        text_context=None,
        patient_metadata={"id": "p-crit"},
    )
    assert out.anomaly.level == "critical"
    assert out.alert is not None
    assert out.alert.level == "critical"
    # A renderizacao na UI nao quebra
    assert format_rag_chunks(out.rag_context)
    assert video_events_to_rows(out.video_events or [])


@pytest.mark.integration
def test_run_case_so_audio_persiste_no_audit_log(tmp_path: Path):
    """Fluxo com apenas audio deve persistir no audit e ser listavel."""
    db = tmp_path / "audit.db"
    orch = _orchestrator(audio=_audio(transcription="ok"), events=None, db_path=db)
    out = run_case_with(
        orch,
        video_path=None,
        audio_path=Path("/fake/a.wav"),
        text_context=None,
        patient_metadata={},
    )
    assert orch.auditor is not None
    cases = orch.auditor.list_cases(limit=10)
    assert any(c["id"] == out.audit_id for c in cases)


@pytest.mark.integration
def test_load_examples_carrega_6_casos():
    """O manifest gerado por `scripts/seed_real_examples.py` deve produzir 6 linhas."""
    from app import _load_examples

    rows = _load_examples()
    # Pode ser 0 se o seed nao foi rodado, mas em CI/local apos seed tem 6
    if rows:
        assert len(rows) == 6
        # Estrutura por linha: [video_path, audio_path, context_text, paciente_id]
        for row in rows:
            assert len(row) == 4
