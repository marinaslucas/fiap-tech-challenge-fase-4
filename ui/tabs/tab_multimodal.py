"""Aba "Multimodal" da UI Gradio.

Combina vıdeo + audio + contexto textual em uma unica execucao do
`Orchestrator.process_case`. Mostra badge de risco, relatorio markdown,
triggers e o contexto RAG consultado.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from ui.components import (
    TRIGGERS_TABLE_HEADERS,
    empty_state,
    format_anomaly_summary,
    format_rag_chunks,
    kpi_grid,
    kpi_tile,
    llm_provider_label,
    progress_breadcrumb,
    risk_badge,
    section_title,
    triggers_to_rows,
)
from ui.limits import validate_audio, validate_video

if TYPE_CHECKING:
    from src.orchestrator import CaseOutput

logger = logging.getLogger(__name__)

CaseRunner = Callable[..., "CaseOutput"]


def render(
    run_case: CaseRunner,
    examples: list[list] | None = None,
    examples_label: str = "Casos pre-carregados",
) -> None:
    """Constroi a aba multimodal.

    Args:
        run_case: callable com assinatura kwargs (`video_path`,
            `audio_path`, `text_context`, `patient_metadata`) que devolve
            um `CaseOutput`. Tipicamente um wrapper em torno de
            `Orchestrator.process_case`.
        examples: opcional, lista de linhas `[video_path, audio_path,
            text_context, paciente_id]` para `gr.Examples`. Quando vazia
            ou `None`, a secao de exemplos nao e renderizada.
        examples_label: titulo da secao de exemplos.
    """
    gr.HTML(progress_breadcrumb(1))

    # ----- 1. ENTRADA (full-width) ----------------------------------------
    with gr.Group(elem_id="caso-clinico-anchor"):
        gr.HTML(
            section_title(
                "Caso clinico",
                "Combine vıdeo, audio e contexto textual. Ao menos uma "
                "modalidade e obrigatoria. O sistema agrega anomalia, "
                "consulta diretrizes (RAG) e gera relatorio clinico em PT-BR.",
            )
        )
        with gr.Row():
            video_input = gr.Video(label="Vıdeo (opcional)", sources=["upload"])
            audio_input = gr.Audio(
                label="Audio (opcional)",
                sources=["upload"],
                type="filepath",
            )
        text_input = gr.Textbox(
            label="Contexto clinico",
            placeholder=("Ex.: Paciente 28 anos, 32 semanas de gestacao, queixa de cefaleia."),
            lines=3,
        )
        patient_id = gr.Textbox(
            label="Identificador da paciente (opcional)",
            placeholder="ex.: paciente-001",
        )
        run_btn = gr.Button("Processar caso", variant="primary", size="lg")

    # ----- 2. EXEMPLOS PRE-CARREGADOS (full-width, aberto por default) ----
    if examples:
        with gr.Accordion(examples_label, open=True):
            gr.HTML(
                '<p style="margin: 0 0 12px 0; color: var(--body-text-color); '
                'font-size: 12px; opacity: 0.75;">'
                "Clique em uma linha para preencher os campos acima."
                "</p>"
            )
            examples_component = gr.Examples(
                examples=examples,
                inputs=[video_input, audio_input, text_input, patient_id],
                label="",
            )
            # Ao clicar em um exemplo, rolar pro card "Caso clinico"
            # (onde os inputs ficam preenchidos).
            examples_component.load_input_event.then(
                fn=None,
                inputs=None,
                outputs=None,
                js=(
                    "() => { "
                    "  const el = document.getElementById('caso-clinico-anchor'); "
                    "  if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'}); "
                    "}"
                ),
            )

    # ----- 3. RESULTADO (full-width, abaixo do form) ----------------------
    with gr.Group(elem_id="resultado-anchor"):
        gr.HTML(section_title("Resultado consolidado"))
        status_html = gr.HTML(
            value=empty_state(
                "Nenhum caso processado ainda.",
                hint=("Preencha ao menos uma modalidade e clique em Processar caso."),
            )
        )
        kpis_html = gr.HTML(value="")
        with gr.Row():
            case_id_box = gr.Textbox(label="Case ID", interactive=False)
            audit_id_box = gr.Textbox(label="Audit ID", interactive=False)

    with gr.Group():
        gr.HTML(
            section_title(
                "Relatorio clinico",
                f"Gerado por: {llm_provider_label()}.",
            )
        )
        with gr.Column(elem_classes="content-box"):
            report_md = gr.Markdown(value="")

    with gr.Group():
        gr.HTML(
            section_title(
                "Resumo de anomalia",
                "Nivel, contagem de triggers e acoes recomendadas.",
            )
        )
        with gr.Column(elem_classes="content-box"):
            anomaly_md = gr.Markdown(value="")

    with gr.Group():
        gr.HTML(
            section_title(
                "Triggers",
                "Regras clinicas e/ou modelo estatistico disparados.",
            )
        )
        triggers_table = gr.Dataframe(
            headers=TRIGGERS_TABLE_HEADERS,
            datatype=["str"] * len(TRIGGERS_TABLE_HEADERS),
            wrap=True,
            interactive=False,
            value=[],
        )

    with gr.Group():
        gr.HTML(
            section_title(
                "Diretrizes consultadas",
                "Trechos recuperados via RAG (Chroma + bge-m3) sobre os PDFs indexados.",
            )
        )
        with gr.Column(elem_classes="content-box"):
            rag_md = gr.Markdown(value="")

    def _on_run(
        video_path: str | None,
        audio_path: str | None,
        text_context: str,
        paciente: str,
        progress=gr.Progress(),  # noqa: B008  # pattern idiomatico do Gradio
    ):
        if not video_path and not audio_path and not (text_context or "").strip():
            return (
                empty_state(
                    "Caso vazio.",
                    hint=(
                        "Forneca ao menos uma modalidade (vıdeo, audio ou contexto "
                        "textual) antes de processar."
                    ),
                ),
                "",
                "",
                "",
                [],
                "",
                "",
                "",
            )

        # Valida limites de tamanho/duracao antes de despachar pro pipeline.
        if video_path:
            v_check = validate_video(Path(video_path))
            if not v_check.ok:
                logger.warning("Video rejeitado na validacao multimodal: %s", v_check.message)
                return (
                    empty_state("Video fora dos limites aceitos.", hint=v_check.message),
                    "", "", "", [], "", "", "",
                )
        if audio_path:
            a_check = validate_audio(Path(audio_path))
            if not a_check.ok:
                logger.warning("Audio rejeitado na validacao multimodal: %s", a_check.message)
                return (
                    empty_state("Audio fora dos limites aceitos.", hint=a_check.message),
                    "", "", "", [], "", "", "",
                )

        def _progress_cb(frac: float, desc: str) -> None:
            progress(frac, desc=desc)

        try:
            output = run_case(
                video_path=Path(video_path) if video_path else None,
                audio_path=Path(audio_path) if audio_path else None,
                text_context=text_context.strip() or None,
                patient_metadata={"id": paciente.strip()} if paciente.strip() else {},
                progress=_progress_cb,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("Falha ao processar caso multimodal na UI: %s", exc)
            return (
                empty_state("Falha ao processar caso.", hint=str(exc)),
                "",
                "",
                "",
                [],
                "",
                "",
                "",
            )

        anomaly = output.anomaly
        modalities: list[str] = []
        if output.video_events is not None:
            modalities.append("video")
        if output.audio_analysis is not None:
            modalities.append("audio")
        if text_context.strip():
            modalities.append("texto")

        status_block = (
            '<div style="display: flex; align-items: center; gap: 12px;">'
            f"{risk_badge(anomaly.level)}"
            '<span style="color: var(--body-text-color); font-size: 13px;">'
            f"Classificacao consolidada do caso ({len(anomaly.triggers)} triggers, "
            f"{len(output.rag_context)} diretrizes consultadas).</span></div>"
        )

        kpis = kpi_grid(
            [
                kpi_tile(
                    "Nivel",
                    {"normal": "Normal", "moderate": "Moderado", "critical": "Critico"}.get(
                        anomaly.level, anomaly.level
                    ),
                    hint="risco final",
                ),
                kpi_tile(
                    "Triggers",
                    str(len(anomaly.triggers)),
                    hint="regras + estatistico",
                ),
                kpi_tile(
                    "Modalidades",
                    str(len(modalities)),
                    hint=", ".join(modalities) if modalities else "nenhuma",
                ),
                kpi_tile(
                    "Diretrizes",
                    str(len(output.rag_context)),
                    hint="chunks RAG",
                ),
            ]
        )

        return (
            status_block,
            kpis,
            output.case_id,
            "" if output.audit_id is None else str(output.audit_id),
            triggers_to_rows(anomaly.triggers),
            format_anomaly_summary(anomaly),
            output.report_markdown,
            format_rag_chunks(output.rag_context),
        )

    run_btn.click(
        fn=lambda: gr.update(interactive=False, value="Processando..."),
        outputs=run_btn,
        queue=False,
        js=(
            "() => { "
            "  const el = document.getElementById('resultado-anchor'); "
            "  if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'}); "
            "}"
        ),
    ).then(
        fn=_on_run,
        inputs=[video_input, audio_input, text_input, patient_id],
        outputs=[
            status_html,
            kpis_html,
            case_id_box,
            audit_id_box,
            triggers_table,
            anomaly_md,
            report_md,
            rag_md,
        ],
        # show_progress="minimal" deixa gr.Progress do _on_run aparecer
        # com texto descritivo ("Processando video..." etc.)
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(interactive=True, value="Processar caso"),
        outputs=run_btn,
        queue=False,
    )
