"""Demo do orquestrador completo.

Roda 3 cenarios sinteticos end-to-end e imprime nivel de risco, triggers
e o trecho inicial do relatorio markdown gerado:

- normal: nenhum sinal clinico
- moderado: sentimento textual negativo + emocao vocal triste
- critico: sangramento em frames consecutivos + termo critico na transcricao

Sem dependencias externas: substitui os pipelines de video e audio por
mocks alinhados aos tipos Pydantic do projeto. Use para validar que o
`Orchestrator` esta enquadrando os triggers, gerando relatorio e
registrando audit corretamente.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock  # noqa: E402

from src.anomaly import AnomalyClassifier  # noqa: E402
from src.audio.types import (  # noqa: E402
    AcousticFeatures,
    AudioAnalysis,
    SentimentResult,
)
from src.audio.types import (
    EmotionScore as AudioEmotionScore,
)
from src.audit import AuditLogger  # noqa: E402
from src.orchestrator import CaseInput, Orchestrator  # noqa: E402
from src.rag.types import Chunk, RetrievalResult  # noqa: E402
from src.video.types import (  # noqa: E402
    BoundingBox,
    Detection,
    VideoEvent,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo_orchestrator")


def _audio(
    *,
    transcription: str = "",
    emotion: AudioEmotionScore | None = None,
    sentiment: SentimentResult | None = None,
) -> AudioAnalysis:
    features = AcousticFeatures(
        duration_s=3.0,
        pitch_mean_hz=200.0,
        pitch_std_hz=12.0,
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


def _video_events(instrument_frames: int, total: int = 6) -> list[VideoEvent]:
    events: list[VideoEvent] = []
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    for i in range(total):
        detections = []
        if i < instrument_frames:
            detections.append(
                Detection(class_id=0, class_name="grasper", confidence=0.9, bbox=bbox)
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


def _orchestrator_for(
    audio: AudioAnalysis | None, video: list[VideoEvent] | None, db_path: Path
) -> Orchestrator:
    video_pipeline = MagicMock()
    video_pipeline.process.return_value = video or []
    audio_pipeline = MagicMock()
    audio_pipeline.process.return_value = audio or _audio()

    chunk = Chunk(
        text="Diretriz: avaliar prontamente queixas de sangramento e dor.",
        source="manual_ms.pdf",
        section="prenatal",
        page=1,
        chunk_id="manual_ms.pdf::1::0",
    )
    retriever = MagicMock()
    retriever.search.return_value = RetrievalResult(chunks=[chunk], scores=[0.9])

    llm = MagicMock()
    llm.is_configured = False  # forca fallback deterministico

    return Orchestrator(
        video_pipeline=video_pipeline,
        audio_pipeline=audio_pipeline,
        classifier=AnomalyClassifier(),
        retriever=retriever,
        llm_client=llm,
        auditor=AuditLogger(db_path=db_path),
    )


def _print_scenario(title: str, orchestrator: Orchestrator, case: CaseInput) -> str:
    out = orchestrator.process_case(case)
    print(f"\n=== {title} ===")
    print(f"case_id      : {out.case_id}")
    print(f"audit_id     : {out.audit_id}")
    print(f"nivel        : {out.anomaly.level}")
    print(f"triggers     : {[t.rule_id for t in out.anomaly.triggers]}")
    if out.alert is not None:
        print(f"alert        : {out.alert.headline()}")
    else:
        print("alert        : (sem alerta)")
    print("relatorio    :")
    for line in out.report_markdown.splitlines()[:8]:
        print(f"  {line}")
    return out.anomaly.level


def main() -> int:
    """Roda 3 cenarios sinteticos e valida niveis."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "audit.sqlite"

        # Cenario 1: normal
        orch = _orchestrator_for(
            audio=_audio(transcription="Tudo bem."),
            video=_video_events(instrument_frames=0, total=3),
            db_path=db,
        )
        level_normal = _print_scenario(
            "Cenario 1 - Normal",
            orch,
            CaseInput(
                video_path=Path("/fake/video.mp4"),
                audio_path=Path("/fake/audio.wav"),
                patient_metadata={"id": "demo-normal"},
            ),
        )

        # Cenario 2: moderado
        orch = _orchestrator_for(
            audio=_audio(
                transcription="Estou me sentindo cansada.",
                sentiment=SentimentResult(label="negative", confidence=0.85),
                emotion=AudioEmotionScore(label="sad", confidence=0.8),
            ),
            video=None,
            db_path=db,
        )
        level_moderado = _print_scenario(
            "Cenario 2 - Moderado",
            orch,
            CaseInput(
                audio_path=Path("/fake/audio.wav"),
                patient_metadata={"id": "demo-moderado"},
            ),
        )

        # Cenario 3: critico
        orch = _orchestrator_for(
            audio=_audio(transcription="Apresentou hemorragia intensa."),
            video=_video_events(instrument_frames=4, total=6),
            db_path=db,
        )
        level_critico = _print_scenario(
            "Cenario 3 - Critico",
            orch,
            CaseInput(
                video_path=Path("/fake/video.mp4"),
                audio_path=Path("/fake/audio.wav"),
                patient_metadata={"id": "demo-critico"},
            ),
        )

        assert level_normal == "normal", f"esperado normal, recebi {level_normal}"
        assert level_moderado == "moderate", f"esperado moderate, recebi {level_moderado}"
        assert level_critico == "critical", f"esperado critical, recebi {level_critico}"
        print("\nTodos os cenarios validados com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
