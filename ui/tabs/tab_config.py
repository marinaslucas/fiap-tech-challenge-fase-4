"""Aba "Configuracoes" da UI Gradio.

Painel read-only que mostra o estado das integracoes externas, toggles
e caminhos detectados pelo `src.config.settings`. Nao permite edicao
em runtime (a config e carregada via Pydantic Settings na inicializacao
do app). Para mudar, edite o `.env` e reinicie.
"""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from src.config.settings import settings
from ui.components import section_title
from ui.theme import (
    ALERT_MODERATE_BG,
    ALERT_NORMAL_BG,
    PRIMARY_SUBTLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def _badge(status: str, label: str) -> str:
    """Pill colorida para o estado: ok/warn/info.

    `info` usa a paleta indigo do tema (azul) em vez de vermelho, pra
    diferenciar visualmente de erro/critico. Estado "cloud ativo" e
    informativo, nao alarme.
    """
    cor_map = {
        "ok": (ALERT_NORMAL_BG, "#14532D"),
        "warn": (ALERT_MODERATE_BG, "#422006"),
        "info": (PRIMARY_SUBTLE, "#1E1B4B"),
    }
    bg, fg = cor_map.get(status, cor_map["info"])
    style = (
        f"background:{bg};color:{fg};padding:2px 10px;border-radius:999px;"
        "font-size:11px;font-weight:600;text-transform:uppercase;"
        "letter-spacing:0.04em;display:inline-block;"
    )
    return f'<span style="{style}">{html.escape(label)}</span>'


def _row(label: str, status_html: str, detail: str) -> str:
    """Linha de uma 'tabela' visual: label + badge + texto descritivo."""
    return (
        '<div style="display: grid; grid-template-columns: 180px 110px 1fr;'
        f"gap: 12px; align-items: center; padding: 10px 0;"
        f' border-bottom: 1px solid {TEXT_SECONDARY}22;">'
        f'<div style="color:{TEXT_PRIMARY};font-weight:600;font-size:13px;">'
        f"{html.escape(label)}</div>"
        f"<div>{status_html}</div>"
        f'<div style="color:{TEXT_SECONDARY};font-size:12px;line-height:1.5;">'
        f"{detail}</div>"
        "</div>"
    )


def _check_azure(key_attr: str, endpoint_attr: str | None = None) -> tuple[str, str]:
    """Avalia uma integracao Azure (key + opcionalmente endpoint).

    Returns:
        Tupla `(status, detail)`. Status em {"ok", "warn"}; detail em texto.
    """
    key = getattr(settings, key_attr).get_secret_value() if hasattr(settings, key_attr) else ""
    if not key:
        return "warn", "Nao configurado"
    if endpoint_attr:
        endpoint = getattr(settings, endpoint_attr)
        if not endpoint:
            return "warn", f"Key presente mas {endpoint_attr} vazio"
    return "ok", "Configurado"


def _check_file(path: Path, label: str) -> tuple[str, str]:
    """Avalia se um arquivo/diretorio existe."""
    if path.exists():
        return "ok", f"{label}: {path.relative_to(settings.project_root)}"
    return "warn", f"{label}: ausente em {path.relative_to(settings.project_root)}"


def _status_panel() -> str:
    """Monta o HTML do painel de status com todas as linhas."""
    linhas: list[str] = []

    # ---- Toggles locais ---------------------------------------------------
    speech_local_ativo = not settings.use_cloud_transcription
    # Emocao facial: GPT-vision multimodal vence quando o deployment esta
    # setado. Caso contrario, cai no FER local.
    vision_cloud_ativo = bool(
        settings.azure_openai_vision_deployment
        and settings.azure_openai_key.get_secret_value()
        and settings.azure_openai_endpoint
    )
    face_local_ativo = not vision_cloud_ativo
    linhas.append(
        _row(
            "Transcricao",
            _badge(
                "ok" if speech_local_ativo else "info", "Local" if speech_local_ativo else "Cloud"
            ),
            (
                "Whisper local (faster-whisper small): multilingue, suporta PT-BR."
                if speech_local_ativo
                else "Azure Speech ativo via USE_CLOUD_TRANSCRIPTION=true."
            ),
        )
    )
    if vision_cloud_ativo:
        emocao_facial_detail = (
            f"Azure OpenAI vision multimodal ({settings.azure_openai_vision_deployment}): "
            "emocao + linguagem corporal numa chamada."
        )
    else:
        emocao_facial_detail = "FER local (Py 3.12); em Py 3.14 retorna lista vazia."
    linhas.append(
        _row(
            "Emocao facial",
            _badge(
                "ok" if face_local_ativo else "info",
                "Local" if face_local_ativo else "Cloud",
            ),
            emocao_facial_detail,
        )
    )

    # Emocao vocal: cloud (GPT-4o multimodal) vence se deployment estiver setado;
    # senao cai no wav2vec2 local (com vies conhecido em PT-BR).
    voice_cloud_ativo = bool(
        settings.azure_openai_audio_deployment
        and settings.azure_openai_key.get_secret_value()
        and settings.azure_openai_endpoint
    )
    linhas.append(
        _row(
            "Emocao vocal",
            _badge(
                "info" if voice_cloud_ativo else "ok",
                "Cloud" if voice_cloud_ativo else "Local",
            ),
            (
                f"Azure OpenAI multimodal ({settings.azure_openai_audio_deployment})."
                if voice_cloud_ativo
                else "wav2vec2 local (superb-er). Vies conhecido em PT-BR; ver relatorio 9.1."
            ),
        )
    )

    # ---- Servicos Azure ---------------------------------------------------
    cases: list[tuple[str, str, str | None, str]] = [
        (
            "Azure Speech",
            "azure_speech_key",
            None,
            "Transcricao cloud. Sem isso, Whisper local responde.",
        ),
        (
            "Azure Language",
            "azure_language_key",
            "azure_language_endpoint",
            "Sentimento + key phrases. Sem isso, regra negative_sentiment nao dispara.",
        ),
        (
            "Azure OpenAI",
            "azure_openai_key",
            "azure_openai_endpoint",
            "Relatorio via GPT-4.1-mini (AI Foundry). Sem isso, fallback deterministico.",
        ),
    ]
    for label, key_attr, endpoint_attr, hint in cases:
        status, msg = _check_azure(key_attr, endpoint_attr)
        badge_label = "Configurado" if status == "ok" else "Inativo"
        detail = msg if status == "ok" else hint
        linhas.append(_row(label, _badge(status, badge_label), detail))

    # ---- Recursos locais --------------------------------------------------
    yolo = settings.yolo_weights_absolute()
    rag = settings.rag_index_absolute()
    audit_db = settings.project_root / "data" / "processed" / "audit.sqlite"

    yolo_status, yolo_detail = _check_file(yolo, "pesos")
    rag_chroma_db = rag / "chroma.sqlite3"
    rag_status = "ok" if rag_chroma_db.exists() else "warn"
    rag_detail = (
        f"indice em {rag.relative_to(settings.project_root)}"
        if rag_chroma_db.exists()
        else "indice ausente. Rode `python scripts/build_rag_index.py`"
    )

    audit_status = "ok" if audit_db.exists() else "warn"
    audit_detail = (
        "banco em data/processed/audit.sqlite"
        if audit_db.exists()
        else "vazio. Sera criado no primeiro caso processado"
    )

    linhas.append(
        _row(
            "YOLO weights",
            _badge(yolo_status, "Detectado" if yolo_status == "ok" else "Faltando"),
            yolo_detail,
        )
    )
    linhas.append(
        _row(
            "RAG index (Chroma)",
            _badge(rag_status, "Indexado" if rag_status == "ok" else "Vazio"),
            rag_detail,
        )
    )
    linhas.append(
        _row(
            "Audit log (SQLite)",
            _badge(audit_status, "Ativo" if audit_status == "ok" else "Vazio"),
            audit_detail,
        )
    )

    # ---- Outros -----------------------------------------------------------
    linhas.append(
        _row(
            "Log level",
            _badge("ok", settings.log_level),
            "Configuravel via LOG_LEVEL no .env (DEBUG, INFO, WARNING, ERROR).",
        )
    )

    return '<div style="margin-top: 8px;">' + "".join(linhas) + "</div>"


def render() -> None:
    """Constroi a aba de configuracoes (read-only)."""
    with gr.Group():
        gr.HTML(
            section_title(
                "Configuracoes detectadas",
                "Estado vivo das integracoes externas, toggles e recursos "
                "locais. Para alterar, edite o .env (ou as variaveis de "
                "ambiente do host) e reinicie o app.",
            )
        )
        with gr.Column(elem_classes="content-box"):
            gr.HTML(value=_status_panel())

    with gr.Group():
        gr.HTML(section_title("Como configurar"))
        with gr.Column(elem_classes="content-box"):
            gr.Markdown(
                "1. Copie `.env.example` para `.env`\n"
                "2. Preencha apenas as chaves Azure que voce tem (todas opcionais)\n"
                "3. Reinicie o app (`python app.py`)\n"
                "4. Volte aqui e confirme que o badge ficou verde\n\n"
                "**Dica:** se quiser pre-baixar todos os modelos antes do primeiro "
                "uso (Whisper, wav2vec2, bge-m3 com total de ~3 GB), rode antes do "
                "`python app.py`:\n\n"
                "```bash\n"
                "python scripts/warmup.py\n"
                "```\n\n"
                "Sem isso, o download acontece lazy na primeira analise de cada "
                "modalidade, travando a UI por 1-2 min por aba."
            )
