"""Aba "Auditoria" da UI Gradio.

Lista os casos registrados pelo `AuditLogger` (SQLite), permite detalhar
um caso pelo `audit_id` e exportar o registro completo em JSON.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import gradio as gr

from ui.components import (
    empty_state,
    kpi_grid,
    kpi_tile,
    risk_badge_inline,
    section_title,
)

logger = logging.getLogger(__name__)

AuditListFn = Callable[[int], list[dict]]
AuditGetFn = Callable[[int], "dict | None"]

LIST_HEADERS: list[str] = ["ID", "Case ID", "Criado em", "Risco", "Modalidades", "Detalhes"]
DETAIL_LINK_HTML: str = (
    '<span class="audit-detail-link">Ver detalhes</span>'
)
DETAIL_COLUMN_INDEX: int = 5
DEFAULT_LIMIT: int = 20


def _compute_refresh(limit: float, list_cases: AuditListFn) -> tuple[list[list], str]:
    """Calcula linhas da tabela e bloco de KPIs para um dado limite.

    Extraido como funcao top-level (em vez de closure) para poder ser
    chamado tanto na inicializacao da aba (popular valor default) quanto
    como callback de eventos (.change, .click, .load).
    """
    try:
        limit_int = max(1, int(limit or DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit_int = DEFAULT_LIMIT
    cases = list_cases(limit_int)
    rows: list[list] = []
    criticos = 0
    moderados = 0
    normais = 0
    for case in cases:
        modalities_raw = case.get("modalities") or "[]"
        try:
            modalities = ", ".join(json.loads(modalities_raw))
        except json.JSONDecodeError:
            modalities = str(modalities_raw)
        level = case.get("risk_level", "normal")
        if level == "critical":
            criticos += 1
        elif level == "moderate":
            moderados += 1
        else:
            normais += 1
        rows.append(
            [
                case.get("id"),
                case.get("case_id", ""),
                case.get("created_at", ""),
                risk_badge_inline(level),
                modalities or "-",
                DETAIL_LINK_HTML,
            ]
        )
    kpis = kpi_grid(
        [
            kpi_tile("Total", str(len(cases)), hint="casos listados"),
            kpi_tile("Normais", str(normais), hint="sem anomalia"),
            kpi_tile("Moderados", str(moderados), hint="atencao"),
            kpi_tile("Criticos", str(criticos), hint="acao imediata"),
        ]
    )
    return rows, kpis


def render(list_cases: AuditListFn, get_case: AuditGetFn) -> None:
    """Constroi a aba de auditoria.

    Args:
        list_cases: callable `(limit) -> list[dict]` (geralmente
            `auditor.list_cases`).
        get_case: callable `(audit_id) -> dict | None` (geralmente
            `auditor.get_case`).
    """
    # Carga inicial: dados ja pre-calculados ao construir a aba para
    # evitar a necessidade de clique inicial. O usuario ja entra na aba
    # vendo a lista populada (e os KPIs).
    initial_rows, initial_kpis = _compute_refresh(DEFAULT_LIMIT, list_cases)

    with gr.Group():
        gr.HTML(
            section_title(
                "Casos registrados",
                "Casos processados pelo orquestrador sao registrados em "
                "SQLite (`data/processed/audit.sqlite`). A lista atualiza "
                "automaticamente ao mudar o limite. Clique em uma linha "
                "para inspecionar o caso.",
            )
        )
        kpis_html = gr.HTML(value=initial_kpis)
        limit_input = gr.Number(
            value=DEFAULT_LIMIT,
            label="Limite (numero maximo de casos)",
            precision=0,
        )
        list_table = gr.Dataframe(
            headers=LIST_HEADERS,
            datatype=["number", "str", "str", "html", "str", "html"],
            wrap=True,
            interactive=False,
            value=initial_rows,
            max_height=520,
            elem_classes="audit-table",
        )

    detail_section = gr.Accordion(
        "Detalhar e exportar um caso", open=False,
        elem_id="audit-detail-anchor",
    )
    with detail_section:
        # ----- Card 1: Audit ID + Buscar detalhe ---------------------------
        with gr.Group():
            gr.HTML(
                section_title(
                    "Buscar detalhe",
                    "Informe um Audit ID da tabela acima para inspecionar o caso "
                    "(relatorio + registro completo em JSON).",
                )
            )
            audit_id_input = gr.Number(value=None, label="Audit ID", precision=0)
            detail_btn = gr.Button("Buscar detalhe", variant="primary")

        status_html = gr.HTML(
            value=empty_state(
                "Nenhum caso selecionado.",
                hint="Informe um Audit ID e clique em Buscar detalhe.",
            )
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Group(elem_classes="audit-detail-pane"):
                    gr.HTML(section_title("Relatorio do caso"))
                    with gr.Column(elem_classes="content-box audit-detail-scroll"):
                        detail_md = gr.Markdown(value="")
            with gr.Column(scale=1):
                with gr.Group(elem_classes="audit-detail-pane"):
                    gr.HTML(section_title("Registro completo (JSON)"))
                    with gr.Column(elem_classes="audit-detail-scroll"):
                        detail_json = gr.JSON(value={})

        # ----- Card 2: Exportar JSON (acao secundaria, separada) -----------
        with gr.Group():
            gr.HTML(
                section_title(
                    "Exportar registro",
                    "Gera o JSON do caso indicado acima para download.",
                )
            )
            export_btn = gr.Button("Exportar JSON", variant="primary")
            download_file = gr.File(label="Arquivo exportado", interactive=False)

    def _on_refresh(limit: float) -> tuple[list[list], str]:
        return _compute_refresh(limit, list_cases)

    def _on_detail(audit_id: float | None) -> tuple[str, str, dict]:
        if audit_id is None:
            return (
                empty_state(
                    "Audit ID nao informado.",
                    hint="Digite o ID numerico do caso na caixa acima.",
                ),
                "",
                {},
            )
        try:
            audit_id_int = int(audit_id)
        except (TypeError, ValueError):
            return (empty_state("Audit ID invalido."), "", {})
        case = get_case(audit_id_int)
        if case is None:
            return (
                empty_state(
                    f"Nenhum caso encontrado para audit_id={audit_id_int}.",
                    hint="Clique em Recarregar para conferir IDs disponiveis.",
                ),
                "",
                {},
            )
        status = (
            '<div style="display: flex; align-items: center; gap: 12px;">'
            f"{risk_badge_inline(case.get('risk_level', 'normal'))}"
            '<span style="color: var(--body-text-color); font-size: 13px;">'
            f"case_id={case.get('case_id')} &middot; "
            f"criado em {case.get('created_at')}</span></div>"
        )
        report_md = case.get("report_md") or "_Sem relatorio armazenado._"
        return status, report_md, case

    def _on_export(audit_id: float | None) -> str | None:
        if audit_id is None:
            return None
        try:
            audit_id_int = int(audit_id)
        except (TypeError, ValueError):
            return None
        case = get_case(audit_id_int)
        if case is None:
            return None
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            prefix=f"audit_{audit_id_int}_",
            encoding="utf-8",
        ) as tmp:
            json.dump(case, tmp, ensure_ascii=False, indent=2, default=str)
            tmp_path = tmp.name
        return str(Path(tmp_path))

    # Auto-refresh ao mudar o Limite
    limit_input.change(
        fn=_on_refresh,
        inputs=[limit_input],
        outputs=[list_table, kpis_html],
        show_progress="minimal",
    )
    detail_btn.click(
        fn=_on_detail,
        inputs=[audit_id_input],
        outputs=[status_html, detail_md, detail_json],
        show_progress="minimal",
    )
    export_btn.click(
        fn=_on_export,
        inputs=[audit_id_input],
        outputs=[download_file],
        show_progress="minimal",
    )

    # Flag passada do callback Python pro JS scroll: True quando o clique
    # foi na coluna "Detalhes" (deve rolar pro accordion), False pra
    # qualquer outra celula (no-op visual).
    should_scroll = gr.State(value=False)

    # Clique na coluna "Detalhes" da tabela: extrai o ID da 1a coluna da
    # linha, preenche o input, dispara o detalhe, abre o accordion e rola
    # ate ele (mesmo padrao de ancora usado na aba Multimodal). Cliques em
    # outras colunas sao no-op (so a coluna explicita aciona).
    def _on_row_select(evt: gr.SelectData, current_rows):
        # current_rows pode chegar como pandas DataFrame (Gradio Dataframe
        # converte internamente) ou lista; tratamos ambos.
        try:
            import pandas as pd
            if isinstance(current_rows, pd.DataFrame):
                n_rows = len(current_rows)
                get_id = lambda r: current_rows.iloc[r, 0]
            else:
                n_rows = len(current_rows or [])
                get_id = lambda r: current_rows[r][0]
        except Exception:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), False,
            )

        noop = (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), False,
        )
        if evt.index is None or n_rows == 0:
            return noop
        # gr.SelectData.index vem como [row, col] em Dataframe.
        if isinstance(evt.index, list):
            row_idx, col_idx = evt.index[0], evt.index[1] if len(evt.index) > 1 else 0
        else:
            row_idx, col_idx = evt.index, DETAIL_COLUMN_INDEX

        # No-op (e sem scroll) se a coluna clicada nao for "Detalhes"
        if col_idx != DETAIL_COLUMN_INDEX:
            return noop
        if row_idx >= n_rows:
            return noop

        audit_id_val = get_id(row_idx)
        status, report_md, raw = _on_detail(audit_id_val)
        return (
            gr.update(value=audit_id_val),
            status,
            report_md,
            raw,
            gr.update(open=True),
            True,
        )

    list_table.select(
        fn=_on_row_select,
        inputs=[list_table],
        outputs=[
            audit_id_input, status_html, detail_md, detail_json,
            detail_section, should_scroll,
        ],
        show_progress="minimal",
    ).then(
        fn=None,
        inputs=[should_scroll],
        outputs=None,
        js=(
            "(scroll) => { "
            "  if (!scroll) return; "
            "  const el = document.getElementById('audit-detail-anchor'); "
            "  if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'}); "
            "}"
        ),
    )
