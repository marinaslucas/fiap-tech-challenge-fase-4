"""Orquestrador multimodal: ponto unico de entrada (`process_case`).

Recebe `CaseInput` com paths de video e/ou audio (e contexto opcional),
roda os pipelines disponiveis, agrega no `AnomalyClassifier`, monta o
contexto RAG, gera o relatorio, dispara o alerta e registra no audit log.

Design:
- Cada modalidade e opcional. Se `video_path` ou `audio_path` for `None`,
  o pipeline correspondente nao roda
- O RAG e consultado a partir das mensagens dos triggers + transcricao,
  e e tolerante a falhas (`try/except` no retrieval)
- Tudo que e externo (LLM, RAG, audit) e injetavel para facilitar testes
  e para a UI poder configurar sem mexer no codigo
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src.alert import Alert, build_alert, dispatch_alert
from src.anomaly.classifier import AnomalyClassifier
from src.anomaly.types import AnomalyResult
from src.audio.pipeline import AudioPipeline
from src.audio.types import AudioAnalysis
from src.audit import AuditLogger
from src.llm.azure_openai import AzureOpenAIClient
from src.rag.retriever import Retriever
from src.rag.types import Chunk
from src.report import generate_report
from src.video.pipeline import VideoPipeline
from src.video.scene_classifier import SceneType
from src.video.types import VideoEvent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.alert import Dispatcher

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

DEFAULT_RAG_TOP_K: int = 6
MAX_QUERY_CHARS: int = 400

# PDFs centrados em cuidado obstetrico/gestacional. Quando o haystack do caso
# nao contem keywords de gestacao, esses documentos sao excluidos da query
# base do RAG pra evitar que o pipeline cite diretrizes pre-natais a
# pacientes nao gestantes.
OBSTETRIC_SOURCES: tuple[str, ...] = (
    "manual_ms_prenatal",
    "ms_gestacao_alto_risco",
    "ms_parto_normal",
    "febrasgo_preeclampsia",
)
PREGNANCY_KEYWORDS: tuple[str, ...] = (
    "gravid", "gesta", "prenatal", "pre-natal", "parto", "puer",
    "amament", "lactant", "concep",
)

# Keywords que indicam contexto cirurgico/laparoscopico. Quando o caso e
# cirurgico e nenhum eixo emocional/violencia ativa, pulamos a query base
# do RAG: nao temos diretrizes cirurgicas indexadas, entao o LLM declara
# "tema fora das diretrizes" em vez de citar diretrizes ginecologicas
# tangenciais.
SURGICAL_KEYWORDS: tuple[str, ...] = (
    "cirurg", "laparoscop", "intraoperat", "instrumental cirurg",
    "grasper", "eletrocaut", "campo cirurg",
)

# Eixos tematicos detectados nos triggers/transcricao pra montar queries focadas.
# Cada entry tem keywords (gatilhos) e um sufixo clinico que orienta o retriever
# pro PDF correto. Multiplos eixos podem ser ativados pra um mesmo caso; cada
# eixo ativo vira uma query separada no `multi_search` (round-robin).
RAG_AXES: dict[str, dict[str, object]] = {
    "saude_mental": {
        "keywords": (
            "ansios", "depress", "medo", "tens", "sofrimento", "angust",
            "choro", "isolam", "desesper", "panico", "suicid", "automutil",
            "transtorno mental", "saude mental", "psicol", "psiquia", "apreens",
        ),
        "suffix": (
            "manejo de transtornos mentais comuns na atencao primaria: "
            "ansiedade, depressao, sofrimento psiquico, encaminhamento para "
            "saude mental"
        ),
        "sources": ("cab34_saude_mental", "manual_ms_prenatal"),
    },
    "violencia": {
        "keywords": (
            "violenc", "agres", "abuso", "ameac", "machucad", "espancad",
            "estupr", "agresor", "agressor", "acolhimento violenc",
        ),
        "suffix": (
            "acolhimento e linha de cuidado a mulheres em situacao de "
            "violencia, notificacao compulsoria, protocolo IST profilaxia"
        ),
        "sources": ("ms_pcdt_ist_violencia",),
    },
    "reprodutivo": {
        "keywords": (
            "gravid", "gesta", "pre-natal", "prenatal", "parto", "puer",
            "contracep", "menstrua", "amament", "lactant", "concep",
            "menopaus", "climater",
        ),
        "suffix": (
            "atencao a saude reprodutiva da mulher: pre-natal, parto, "
            "puerperio, contracepcao, planejamento reprodutivo"
        ),
        "sources": (
            "manual_ms_prenatal", "ms_gestacao_alto_risco",
            "cab26_saude_sexual_reprodutiva", "ms_parto_normal",
            "febrasgo_preeclampsia",
        ),
    },
    "rastreio": {
        "keywords": (
            "nodulo", "mamograf", "papanicolau", "citologi", "rastrei",
            "preventiv", "tumor", "cancer", "neoplasia", "hpv", "colposcop",
        ),
        "suffix": (
            "rastreamento e deteccao precoce: cancer de mama, cancer de "
            "colo do utero, citologia, exames preventivos"
        ),
        "sources": ("inca_cancer_mama", "inca_cancer_colo_utero"),
    },
}


class CaseInput(BaseModel):
    """Entrada do orquestrador para um caso clinico completo."""

    video_path: Path | None = None
    audio_path: Path | None = None
    text_context: str | None = None
    patient_metadata: dict = Field(default_factory=dict)


class CaseOutput(BaseModel):
    """Saida consolidada do orquestrador.

    Contem todas as analises feitas para o caso, prontas para a UI,
    para o relatorio e para a auditoria.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: str
    video_events: list[VideoEvent] | None = None
    audio_analysis: AudioAnalysis | None = None
    rag_context: list[Chunk] = Field(default_factory=list)
    anomaly: AnomalyResult
    report_markdown: str
    alert: Alert | None = None
    audit_id: int | None = None
    scene_type: SceneType | None = None


class Orchestrator:
    """Cola entre pipelines, classificador, RAG, LLM, audit e alerta."""

    def __init__(
        self,
        *,
        video_pipeline: VideoPipeline | None = None,
        audio_pipeline: AudioPipeline | None = None,
        classifier: AnomalyClassifier | None = None,
        retriever: Retriever | None = None,
        llm_client: AzureOpenAIClient | None = None,
        auditor: AuditLogger | None = None,
        alert_dispatchers: Sequence[Dispatcher] | None = None,
        rag_top_k: int = DEFAULT_RAG_TOP_K,
    ) -> None:
        """Configura o orquestrador.

        Args:
            video_pipeline: pipeline de video; default `VideoPipeline()`.
            audio_pipeline: pipeline de audio; default `AudioPipeline()`.
            classifier: classificador final; default `AnomalyClassifier()`.
            retriever: retriever RAG; default `Retriever()`. Se a indexacao
                falhar, o retriever sera ignorado em runtime.
            llm_client: cliente LLM; default `AzureOpenAIClient()`.
            auditor: logger SQLite; default `AuditLogger()`.
            alert_dispatchers: lista de dispatchers; default `[_log_dispatcher]`.
            rag_top_k: quantos chunks recuperar do RAG.
        """
        self.video_pipeline: VideoPipeline = video_pipeline or VideoPipeline()
        self.audio_pipeline: AudioPipeline = audio_pipeline or AudioPipeline()
        self.classifier: AnomalyClassifier = classifier or AnomalyClassifier()
        self.retriever: Retriever | None = retriever if retriever is not None else None
        self.llm_client: AzureOpenAIClient = llm_client or AzureOpenAIClient()
        self.auditor: AuditLogger | None = auditor
        self.alert_dispatchers: list[Dispatcher] | None = (
            list(alert_dispatchers) if alert_dispatchers else None
        )
        self.rag_top_k: int = rag_top_k

    def process_case(
        self,
        case: CaseInput,
        progress: ProgressCallback | None = None,
    ) -> CaseOutput:
        """Pipeline completo para um caso multimodal.

        Args:
            case: entrada com paths e contexto opcional.
            progress: callback opcional `(frac, desc)` com `frac` em [0, 1]
                e `desc` string curta. Util pra integrar com `gr.Progress`
                na UI Gradio. Default `None` (sem reportes).

        Returns:
            `CaseOutput` com todas as analises, relatorio, alerta e id de audit.
        """
        case_id = self._new_case_id()
        logger.info("Iniciando caso %s", case_id)

        if progress is not None:
            progress(0.05, "Processando video...")
        video_events = self._run_video(case.video_path)
        scene_type: SceneType | None = (
            self.video_pipeline.last_scene_type
            if case.video_path is not None and video_events is not None
            else None
        )

        if progress is not None:
            progress(0.45, "Processando audio...")
        audio_analysis = self._run_audio(case.audio_path)

        if progress is not None:
            progress(0.65, "Detectando anomalia...")
        anomaly = self.classifier.classify(video_events=video_events, audio_analysis=audio_analysis)

        if progress is not None:
            progress(0.70, "Recuperando diretrizes (RAG)...")
        rag_context = self._retrieve_context(case, audio_analysis, anomaly)

        if progress is not None:
            progress(0.80, "Gerando relatorio com LLM...")
        report_markdown = generate_report(
            anomaly=anomaly,
            audio_analysis=audio_analysis,
            video_events=video_events,
            rag_chunks=rag_context,
            patient_metadata=case.patient_metadata,
            scene_type=scene_type,
            llm_client=self.llm_client,
        )

        if progress is not None:
            progress(0.95, "Registrando auditoria...")
        alert = build_alert(case_id=case_id, anomaly=anomaly)
        dispatch_alert(alert, self.alert_dispatchers)

        modalities: list[str] = []
        if video_events is not None:
            modalities.append("video")
        if audio_analysis is not None:
            modalities.append("audio")
        if case.text_context:
            modalities.append("text")

        audit_id: int | None = None
        if self.auditor is not None:
            audit_id = self.auditor.log_case(
                case_id=case_id,
                anomaly=anomaly,
                modalities=modalities,
                report_md=report_markdown,
                metadata={
                    "patient": case.patient_metadata,
                    "text_context": case.text_context,
                },
            )

        return CaseOutput(
            case_id=case_id,
            video_events=video_events,
            audio_analysis=audio_analysis,
            rag_context=rag_context,
            anomaly=anomaly,
            report_markdown=report_markdown,
            alert=alert,
            audit_id=audit_id,
            scene_type=scene_type,
        )

    def _new_case_id(self) -> str:
        """Gera um identificador unico curto para o caso."""
        return f"case-{uuid.uuid4().hex[:12]}"

    def _run_video(self, video_path: Path | None) -> list[VideoEvent] | None:
        """Roda o pipeline de video se houver path; senao retorna `None`."""
        if video_path is None:
            return None
        try:
            return self.video_pipeline.process(video_path)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Falha no pipeline de video: %s", exc)
            return None

    def _run_audio(self, audio_path: Path | None) -> AudioAnalysis | None:
        """Roda o pipeline de audio se houver path; senao retorna `None`."""
        if audio_path is None:
            return None
        try:
            return self.audio_pipeline.process(audio_path)
        except FileNotFoundError as exc:
            logger.warning("Falha no pipeline de audio: %s", exc)
            return None

    def _retrieve_context(
        self,
        case: CaseInput,
        audio_analysis: AudioAnalysis | None,
        anomaly: AnomalyResult,
    ) -> list[Chunk]:
        """Monta queries por eixo tematico e funde resultados via multi_search.

        Em vez de uma unica query unificada (que mistura sinais e deixa o
        ranking dominado pelo PDF mais volumoso), monta uma query base
        clinica + queries especificas pros eixos detectados (saude mental,
        violencia, reprodutivo, rastreio). Cada eixo vira uma query focada,
        os resultados sao deduplicados e interleaved no `multi_search`.
        """
        if self.retriever is None:
            return []

        base_parts: list[str] = []
        if case.text_context:
            base_parts.append(case.text_context)
        if audio_analysis is not None and audio_analysis.transcription:
            base_parts.append(audio_analysis.transcription)
        trigger_text = " ".join(t.message for t in anomaly.triggers)

        base_query = " ".join(base_parts).strip()
        if not base_query and not trigger_text:
            return []

        # Texto a inspecionar pra detectar eixos. Inclui contexto, transcricao
        # e mensagens de trigger (todos lowercase pra match com keywords).
        haystack = " ".join([base_query, trigger_text]).lower()

        # Sempre roda a query "clinica" base (contexto + transcricao + triggers)
        # sem filtro de source: ela captura tema central do caso. Quando o
        # caso nao tem nenhum sinal de gestacao no haystack (fala da paciente,
        # contexto clinico, triggers), excluimos os PDFs obstetricos da query
        # base pra evitar que dominem o ranking em pacientes nao gestantes.
        is_pregnancy_case = any(kw in haystack for kw in PREGNANCY_KEYWORDS)
        is_surgical_case = any(kw in haystack for kw in SURGICAL_KEYWORDS)
        clinical_query = " ".join([base_query, trigger_text]).strip()[:MAX_QUERY_CHARS]
        queries: list[str] = [clinical_query]
        sources_per_query: list[tuple[str, ...] | None] = [None]
        excluded_sources_per_query: list[tuple[str, ...] | None] = [
            None if is_pregnancy_case else OBSTETRIC_SOURCES,
        ]

        # Detecta eixos ativados pelas keywords e adiciona queries focadas
        # com allowlist de sources pra garantir representacao do PDF correto.
        # Query focada usa SO o sufixo do eixo (sem context prefix), evitando
        # que o embedding fique parecido demais com a query clinica.
        activated: list[str] = []
        for axis_name, axis in RAG_AXES.items():
            keywords: tuple[str, ...] = axis["keywords"]  # type: ignore[assignment]
            if any(kw in haystack for kw in keywords):
                suffix: str = axis["suffix"]  # type: ignore[assignment]
                sources: tuple[str, ...] = axis["sources"]  # type: ignore[assignment]
                queries.append(suffix[:MAX_QUERY_CHARS])
                sources_per_query.append(sources)
                # Queries focadas ja tem allowlist; nao precisam de denylist.
                excluded_sources_per_query.append(None)
                activated.append(axis_name)

        # Em caso cirurgico sem sinal humano (sem eixo emocional, violencia
        # ou reprodutivo ativados), nao temos diretrizes aplicaveis. Pular
        # a query base evita que o RAG retorne chunks tangenciais (ex.: 1
        # chunk de cancer de colo uterino que o LLM tenta "fazer caber"
        # alucinando relacao com a cirurgia abdominal). Quando os chunks
        # vem vazios, o LLM declara "tema fora das diretrizes indexadas"
        # conforme regra 5 do prompt.
        if is_surgical_case and not activated:
            queries = []
            sources_per_query = []
            excluded_sources_per_query = []

        logger.info(
            "RAG multi-query: %d queries (eixos ativados: %s, gestacional: %s, cirurgico: %s)",
            len(queries),
            activated or ["nenhum, so base"],
            is_pregnancy_case,
            is_surgical_case,
        )

        if not queries:
            # Sem queries -> RAG vazio. LLM tratara como "tema fora".
            return []

        try:
            result = self.retriever.multi_search(
                queries,
                top_k_per_query=3,
                total_k=self.rag_top_k,
                sources_per_query=sources_per_query,
                excluded_sources_per_query=excluded_sources_per_query,
            )
        except Exception as exc:  # noqa: BLE001 - RAG pode falhar por falta de indice
            logger.warning("RAG retrieval falhou: %s", exc)
            return []
        return list(result.chunks)


def process_case(case: CaseInput) -> CaseOutput:
    """Atalho de conveniencia: instancia um `Orchestrator` padrao e processa."""
    return Orchestrator().process_case(case)
