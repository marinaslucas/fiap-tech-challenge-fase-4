"""Entrypoint Gradio Blocks do medica-ia-multimodal.

Monta o app com tema dark indigo, CSS customizado e 4 abas (Vıdeo, Audio,
Multimodal, Auditoria). Os pipelines pesados (Whisper, YOLO, MediaPipe)
sao instanciados sob demanda na primeira chamada de cada aba, evitando
download de modelos no startup.

Uso:
    python app.py [--share] [--server-port 7860]

Sem args sobe local em http://127.0.0.1:7860.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from src.audit import AuditLogger
from src.config.settings import settings
from src.orchestrator import CaseInput, Orchestrator
from ui.components import footer_html, header_html
from ui.tabs import tab_audio, tab_audit, tab_config, tab_multimodal, tab_video
from ui.theme import get_custom_css, get_theme

if TYPE_CHECKING:
    from src.audio.types import AudioAnalysis
    from src.orchestrator import CaseOutput

logger = logging.getLogger(__name__)

EXAMPLES_DIR: Path = Path("data/examples")


def _load_examples() -> list[list]:
    """Le `data/examples/manifest.json` e converte em linhas para `gr.Examples`.

    Returns:
        Lista de `[video_path, audio_path, text_context, paciente_id]`
        com paths absolutos (Gradio precisa pra resolver). Lista vazia se
        manifest nao existir ou parsing falhar.
    """
    manifest_path = settings.project_root / EXAMPLES_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Manifest de exemplos invalido: %s", exc)
        return []

    rows: list[list] = []
    for caso in manifest.get("casos", []):
        video_path = caso.get("video_path")
        audio_path = caso.get("audio_path")
        rows.append(
            [
                str(settings.project_root / video_path) if video_path else None,
                str(settings.project_root / audio_path) if audio_path else None,
                caso.get("context_text", ""),
                (caso.get("patient_metadata") or {}).get("id", ""),
            ]
        )
    return rows


def _build_orchestrator() -> Orchestrator:
    """Instancia o orquestrador padrao com retriever opcional.

    Tenta criar o `Retriever` apontando para o ındice Chroma persistido em
    `data/processed/chroma`. Se nao houver ındice ou o pacote falhar, segue
    sem RAG (o orquestrador retorna `[]` no contexto).
    """
    retriever = None
    try:
        from src.rag.retriever import Retriever

        retriever = Retriever()
    except Exception as exc:  # noqa: BLE001 - RAG e opcional para a UI
        logger.warning("RAG indisponivel na inicializacao: %s", exc)

    auditor = AuditLogger()
    return Orchestrator(retriever=retriever, auditor=auditor)


def run_case_with(
    orchestrator: Orchestrator,
    *,
    video_path: Path | None,
    audio_path: Path | None,
    text_context: str | None,
    patient_metadata: dict,
    progress=None,
) -> CaseOutput:
    """Wrapper testavel de `Orchestrator.process_case`.

    Existe como funcao top-level (em vez de closure dentro de `build_app`)
    para que testes de integracao possam exercer o mesmo caminho da UI.

    Args:
        progress: callable opcional `(frac, desc)` repassado pro orquestrador
            (geralmente `gr.Progress()` da UI Gradio).
    """
    case = CaseInput(
        video_path=video_path,
        audio_path=audio_path,
        text_context=text_context,
        patient_metadata=patient_metadata,
    )
    return orchestrator.process_case(case, progress=progress)


def build_app(orchestrator: Orchestrator | None = None) -> gr.Blocks:
    """Monta o `gr.Blocks` com tema, CSS, header e as 4 abas.

    Args:
        orchestrator: instancia compartilhada. Default cria via
            `_build_orchestrator`. Injetavel para testes.

    Returns:
        Bloco Gradio pronto para `.launch()`.
    """
    orch = orchestrator or _build_orchestrator()

    def _process_audio(path: Path, progress=None) -> AudioAnalysis:
        return orch.audio_pipeline.process(path, progress=progress)

    def _run_case(
        *,
        video_path: Path | None,
        audio_path: Path | None,
        text_context: str | None,
        patient_metadata: dict,
        progress=None,
    ) -> CaseOutput:
        return run_case_with(
            orch,
            video_path=video_path,
            audio_path=audio_path,
            text_context=text_context,
            patient_metadata=patient_metadata,
            progress=progress,
        )

    auditor = orch.auditor or AuditLogger()

    with gr.Blocks(
        theme=get_theme(),
        css=get_custom_css(),
        title="medica-ia-multimodal",
    ) as app:
        gr.HTML(header_html("Sistema online"))

        with gr.Tabs():
            with gr.TabItem("Vıdeo"):
                tab_video.render(video_pipeline=orch.video_pipeline)
            with gr.TabItem("Audio"):
                tab_audio.render(process_audio=_process_audio)
            with gr.TabItem("Multimodal"):
                tab_multimodal.render(
                    run_case=_run_case,
                    examples=_load_examples(),
                )
            with gr.TabItem("Auditoria"):
                tab_audit.render(
                    list_cases=auditor.list_cases,
                    get_case=auditor.get_case,
                )
            with gr.TabItem("Configuracoes", elem_classes=["tab-config-end"]):
                tab_config.render()

        gr.HTML(
            footer_html(
                version="0.1.0",
                repo_url="https://github.com/joalissonborges/medica-ia-multimodal",
            )
        )

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Le os argumentos de linha de comando.

    Defaults respeitam env vars GRADIO_SERVER_NAME e GRADIO_SERVER_PORT
    quando setadas (Dockerfile usa 0.0.0.0 pra expor a aplicacao fora
    do container). CLI args ainda sobrescrevem se passados explicitamente.
    """
    default_server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    default_server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Entrypoint Gradio do medica-ia-multimodal")
    parser.add_argument("--share", action="store_true", help="Cria URL publico via Gradio")
    parser.add_argument(
        "--server-name",
        default=default_server_name,
        help=f"Host para servir (default {default_server_name}, "
        "respeita env GRADIO_SERVER_NAME)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=default_server_port,
        help=f"Porta TCP (default {default_server_port}, "
        "respeita env GRADIO_SERVER_PORT)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entrypoint CLI: sobe o servidor Gradio."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    app = build_app()
    app.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_api=False,
    )


if __name__ == "__main__":
    main()
