"""Helpers visuais reutilizaveis pelas abas Gradio.

Consome os tokens de `ui/theme.py` (paleta dark indigo) para manter o
visual consistente entre vıdeo, audio, multimodal e auditoria. Os helpers
produzem strings HTML ou estruturas tabulares prontas para os componentes
Gradio (`gr.HTML`, `gr.Dataframe`, `gr.Markdown`).
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from ui.theme import (
    ALERT_CRITICAL_BG,
    ALERT_CRITICAL_FG,
    ALERT_MODERATE_BG,
    ALERT_MODERATE_FG,
    ALERT_NORMAL_BG,
    ALERT_NORMAL_FG,
    BG_SURFACE,
    BORDER,
    PRIMARY,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
)

if TYPE_CHECKING:
    from src.anomaly.types import AnomalyResult, RiskLevel, Trigger
    from src.audio.types import AcousticFeatures, AudioAnalysis
    from src.rag.types import Chunk
    from src.video.types import VideoEvent

# Rotulos PT-BR para os niveis (interface clinica).
RISK_LEVEL_LABELS: dict[str, str] = {
    "normal": "Normal",
    "moderate": "Moderado",
    "critical": "Critico",
}

# Mapa nivel -> classe CSS (declaradas em `ui/theme.py:get_custom_css`).
RISK_LEVEL_CSS_CLASS: dict[str, str] = {
    "normal": "alert-normal",
    "moderate": "alert-moderate",
    "critical": "alert-critical",
}

# Cores inline (usadas quando a classe CSS nao for aplicavel, ex.: dentro
# de uma celula `gr.Dataframe`).
RISK_LEVEL_INLINE_COLORS: dict[str, tuple[str, str]] = {
    "normal": (ALERT_NORMAL_BG, ALERT_NORMAL_FG),
    "moderate": (ALERT_MODERATE_BG, ALERT_MODERATE_FG),
    "critical": (ALERT_CRITICAL_BG, ALERT_CRITICAL_FG),
}


def header_html(version_label: str = "Sistema online") -> str:
    """Header do app com logo estilizado + indicador de status em pill.

    Args:
        version_label: texto curto exibido na pill de status (ex.: "Sistema online").

    Returns:
        Bloco HTML que consome `.app-header`, `.logo`, `.status-pill`, `.dot`
        definidos em `ui/theme.py`.
    """
    safe_label = html.escape(version_label)
    return (
        '<div class="app-header">'
        '<div class="logo">medica-<span class="accent">ia</span>-multimodal</div>'
        f'<div class="status-pill"><span class="dot"></span>{safe_label}</div>'
        "</div>"
    )


def risk_badge(level: RiskLevel | str) -> str:
    """Retorna o HTML de uma badge de risco usando classes CSS do tema.

    Args:
        level: `"normal"`, `"moderate"` ou `"critical"`.

    Returns:
        `<span class="alert-...">Rotulo</span>` pronto para `gr.HTML`.
    """
    css_class = RISK_LEVEL_CSS_CLASS.get(level, "alert-normal")
    label = RISK_LEVEL_LABELS.get(level, str(level).title())
    return f'<span class="{css_class}">{html.escape(label)}</span>'


SCENE_TYPE_LABELS: dict[str, tuple[str, str]] = {
    "cirurgia":      ("Cirurgia",    "#ef4444"),  # vermelho
    "consulta":      ("Consulta",    "#3b82f6"),  # azul
    "misto":         ("Misto",       "#a855f7"),  # roxo
    "desconhecido":  ("Indefinido",  "#64748b"),  # cinza
}


def scene_type_badge(scene_type_value: str) -> str:
    """Badge colorida pra `SceneType`.

    Args:
        scene_type_value: string do enum (`"cirurgia"`, `"consulta"`, etc.).

    Returns:
        `<span style="...">Rotulo</span>` pronto pra `gr.HTML`.
    """
    label, color = SCENE_TYPE_LABELS.get(
        scene_type_value, ("Indefinido", "#64748b")
    )
    style = (
        f"background-color: {color}22; color: {color}; padding: 3px 10px; "
        "border-radius: 4px; font-size: 11px; font-weight: 600; "
        f"border: 1px solid {color}55;"
    )
    return f'<span style="{style}">{html.escape(label)}</span>'


_DETECTION_PALETTE_BGR: dict[str, tuple[int, int, int]] = {
    "grasper":               (0, 255, 0),     # verde
    "l_hook_electrocautery": (255, 0, 255),   # magenta
    "blood":                 (0, 0, 255),     # vermelho
}
_DETECTION_DEFAULT_COLOR_BGR: tuple[int, int, int] = (255, 255, 0)  # ciano


def _render_detection_overlay(frame, detections):
    """Desenha bboxes + labels sobre um frame BGR (modifica in-place)."""
    import cv2

    for det in detections:
        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
        color = _DETECTION_PALETTE_BGR.get(det.class_name, _DETECTION_DEFAULT_COLOR_BGR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )


def build_detection_thumbnails(
    video_path: str, events: list, max_thumbs: int = 4,
):
    """Pega os top N frames com mais deteccoes e desenha bboxes sobrepostas.

    Util pra mostrar visualmente o que o detector YOLO viu em um grid.
    Retorna lista vazia se nao houver deteccoes em nenhum frame.

    Args:
        video_path: caminho absoluto do arquivo de video.
        events: lista de `VideoEvent` (do pipeline.process()).
        max_thumbs: numero maximo de frames a retornar (default 4 = grid 2x2).

    Returns:
        Lista de frames numpy RGB com bboxes desenhadas, ordenados
        temporalmente pelo frame_index. Vazia se nao ha deteccoes ou se
        a leitura do video falhar.
    """
    import cv2

    events_with_dets = [e for e in events if e.detections]
    if not events_with_dets:
        return []

    # Top N por numero de deteccoes, depois reordena temporalmente
    top_events = sorted(
        events_with_dets,
        key=lambda e: len(e.detections),
        reverse=True,
    )[:max_thumbs]
    top_events.sort(key=lambda e: e.frame_index)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    thumbnails = []
    try:
        for event in top_events:
            cap.set(cv2.CAP_PROP_POS_FRAMES, event.frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            _render_detection_overlay(frame, event.detections)
            thumbnails.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    return thumbnails


def emotion_source_label(classifier_class_name: str) -> str:
    """Mapeia nome da classe do classifier de emocao para rotulo amigavel.

    Args:
        classifier_class_name: ex: `"AzureOpenAIVisionEmotion"`, `"FacialEmotionDetector"`.

    Returns:
        String pt-BR pra exibir na UI.
    """
    return {
        "AzureOpenAIVisionEmotion": "Azure GPT-vision",
        "FacialEmotionDetector":    "FER local",
    }.get(classifier_class_name, classifier_class_name)


def risk_badge_inline(level: RiskLevel | str) -> str:
    """Versao com estilo inline para contextos onde o CSS global nao se aplica.

    Args:
        level: `"normal"`, `"moderate"` ou `"critical"`.

    Returns:
        `<span style="...">Rotulo</span>` pronto para celulas de dataframe.
    """
    bg, fg = RISK_LEVEL_INLINE_COLORS.get(level, RISK_LEVEL_INLINE_COLORS["normal"])
    label = RISK_LEVEL_LABELS.get(level, str(level).title())
    style = (
        f"background-color: {bg}; color: {fg}; padding: 3px 10px; "
        "border-radius: 4px; font-size: 11px; font-weight: 500;"
    )
    return f'<span style="{style}">{html.escape(label)}</span>'


def triggers_to_rows(triggers: list[Trigger]) -> list[list[str]]:
    """Converte triggers em linhas para `gr.Dataframe`.

    Returns:
        Lista de `[rule_id, nivel, origem, mensagem]` em PT-BR.
    """
    rows: list[list[str]] = []
    for trigger in triggers:
        rows.append(
            [
                trigger.rule_id,
                RISK_LEVEL_LABELS.get(trigger.level, trigger.level),
                trigger.source,
                trigger.message,
            ]
        )
    return rows


TRIGGERS_TABLE_HEADERS: list[str] = ["Regra", "Nivel", "Origem", "Mensagem"]


def video_events_to_rows(events: list[VideoEvent], max_rows: int = 30) -> list[list[str]]:
    """Resume eventos de vıdeo em linhas tabulares (1 linha por frame).

    Util pra inspeção granular. Use `video_events_to_windowed_rows` para
    agregar em janelas temporais (melhor pra demos longas).

    Para cada frame mostra: indice, timestamp em segundos, numero de
    deteccoes, classes detectadas (concatenadas) e emocao facial (se houver).
    """
    rows: list[list[str]] = []
    for event in events[:max_rows]:
        classes = ", ".join(f"{d.class_name} ({d.confidence:.0%})" for d in event.detections) or "-"
        emotion = (
            f"{event.facial_emotion.label_pt} ({event.facial_emotion.confidence:.0%})"
            if event.facial_emotion is not None
            else "-"
        )
        rows.append(
            [
                str(event.frame_index),
                f"{event.timestamp_ms / 1000:.2f}s",
                str(len(event.detections)),
                classes,
                emotion,
            ]
        )
    return rows


def video_events_to_windowed_rows(
    events: list[VideoEvent],
    window_seconds: float = 5.0,
) -> list[list[str]]:
    """Agrega eventos em janelas temporais pra demos longas.

    Para cada janela retorna: range de tempo, n frames, n deteccoes,
    classes predominantes (com confianca media), emocao predominante,
    linguagem corporal predominante. Vazio quando nenhum frame caiu
    na janela.

    Args:
        events: lista do pipeline.process().
        window_seconds: duracao de cada janela em segundos (default 5).

    Returns:
        Lista de listas no formato `gr.Dataframe` com 6 colunas (ver
        `VIDEO_WINDOWS_HEADERS`).
    """
    from collections import Counter, defaultdict

    if not events:
        return []

    window_ms = int(window_seconds * 1000)
    buckets: dict[int, list[VideoEvent]] = defaultdict(list)
    for event in events:
        bucket_key = event.timestamp_ms // window_ms
        buckets[bucket_key].append(event)

    rows: list[list[str]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        start_s = key * window_seconds
        end_s = (key + 1) * window_seconds - 1
        n_frames = len(bucket)
        total_dets = sum(len(e.detections) for e in bucket)

        # Classes com confianca media por classe
        class_confs: dict[str, list[float]] = defaultdict(list)
        for e in bucket:
            for d in e.detections:
                class_confs[d.class_name].append(d.confidence)
        if class_confs:
            classes_str = ", ".join(
                f"{name} ({sum(confs) / len(confs):.0%})"
                for name, confs in sorted(class_confs.items())
            )
        else:
            classes_str = "-"

        # Emocao predominante (moda)
        emotion_labels = [
            e.facial_emotion.label_pt for e in bucket if e.facial_emotion is not None
        ]
        if emotion_labels:
            label, count = Counter(emotion_labels).most_common(1)[0]
            avg_conf = sum(
                e.facial_emotion.confidence for e in bucket
                if e.facial_emotion is not None and e.facial_emotion.label_pt == label
            ) / count
            emotion_str = f"{label} ({avg_conf:.0%})"
        else:
            emotion_str = "-"

        # Linguagem corporal predominante (vinda do GPT-vision via EmotionScore)
        body_labels = [
            e.facial_emotion.body_language for e in bucket
            if e.facial_emotion is not None and e.facial_emotion.body_language
        ]
        if body_labels:
            top = Counter(body_labels).most_common()
            n_body = len(body_labels)
            if len(top) == 1 or top[0][1] >= n_body * 0.7:
                body_str = f"{top[0][0]} ({top[0][1]}/{n_body})"
            else:
                body_str = (
                    f"{top[0][0]} ({top[0][1]}/{n_body}), "
                    f"{top[1][0]} ({top[1][1]}/{n_body})"
                )
        else:
            body_str = "-"

        rows.append(
            [
                f"{start_s:.0f}-{end_s:.0f}s",
                f"{n_frames}",
                str(total_dets),
                classes_str,
                emotion_str,
                body_str,
            ]
        )
    return rows


VIDEO_EVENTS_HEADERS: list[str] = [
    "Frame",
    "Tempo",
    "Deteccoes",
    "Classes",
    "Emocao facial",
]


VIDEO_WINDOWS_HEADERS: list[str] = [
    "Janela",
    "Frames",
    "Deteccoes",
    "Classes",
    "Emocao predominante",
    "Linguagem corporal predominante",
]


def format_acoustic_features(features: AcousticFeatures | None) -> str:
    """Formata features acusticas como markdown legivel.

    Args:
        features: saida do `extract_features` do pipeline de audio.

    Returns:
        String markdown ou mensagem indicando ausencia de features.
    """
    if features is None:
        return "_Features acusticas indisponiveis._"
    return (
        "| Metrica | Valor |\n"
        "|---|---|\n"
        f"| Duracao | {features.duration_s:.2f} s |\n"
        f"| Pitch medio | {features.pitch_mean_hz:.1f} Hz |\n"
        f"| Pitch std | {features.pitch_std_hz:.1f} Hz |\n"
        f"| Energia (RMS) | {features.energy_rms:.4f} |\n"
        f"| Zero crossing rate | {features.zero_crossing_rate:.4f} |\n"
        f"| Jitter | {features.jitter:.4f} |\n"
        f"| Shimmer | {features.shimmer:.4f} |\n"
    )


def format_audio_summary(analysis: AudioAnalysis | None) -> str:
    """Resume `AudioAnalysis` em markdown para a aba de audio.

    Inclui transcricao, emocao, sentimento e key phrases. Cada bloco vira
    uma secao curta. Quando algum campo nao esta disponivel, mostra um
    placeholder discreto.
    """
    if analysis is None:
        return "_Sem analise de audio disponivel._"

    lines: list[str] = []
    transcription = analysis.transcription.strip() or "_Sem transcricao detectada._"
    lines.append("### Transcricao")
    lines.append(transcription)
    lines.append("")

    if analysis.emotion is not None:
        emo = analysis.emotion
        lines.append(f"**Emocao vocal:** {emo.label_pt} ({emo.confidence:.0%})")
    if analysis.sentiment is not None:
        sent = analysis.sentiment
        if sent.label == "mixed":
            pos = sent.scores.get("positive", 0.0)
            neg = sent.scores.get("negative", 0.0)
            lines.append(
                f"**Sentimento do texto:** {sent.label} "
                f"(pos: {pos:.0%} / neg: {neg:.0%})"
            )
        else:
            lines.append(
                f"**Sentimento do texto:** {sent.label} ({sent.confidence:.0%})"
            )
    if analysis.key_phrases:
        phrases = ", ".join(analysis.key_phrases[:8])
        lines.append(f"**Frases-chave:** {phrases}")
    return "\n".join(lines)


def format_rag_chunks(chunks: list[Chunk]) -> str:
    """Formata os trechos do RAG como uma lista markdown auditavel.

    Cada chunk vira um item com `source`, `section`, `page` e o trecho.
    """
    if not chunks:
        return "_Sem diretrizes recuperadas no contexto._"
    lines: list[str] = ["### Diretrizes consultadas"]
    for chunk in chunks:
        meta_parts: list[str] = [chunk.source]
        if chunk.section:
            meta_parts.append(chunk.section)
        if chunk.page is not None:
            meta_parts.append(f"p. {chunk.page}")
        meta = " | ".join(meta_parts)
        snippet = chunk.text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        lines.append(f"- **{meta}**  \n  {snippet}")
    return "\n".join(lines)


def format_anomaly_summary(anomaly: AnomalyResult) -> str:
    """Monta um resumo curto da anomalia em markdown.

    Inclui o nivel, contagem de triggers e a `explanation` ja em PT-BR.
    """
    label = RISK_LEVEL_LABELS.get(anomaly.level, anomaly.level)
    pieces: list[str] = [
        f"**Nivel:** {label}",
        f"**Triggers:** {len(anomaly.triggers)}",
    ]
    if anomaly.explanation:
        pieces.append(anomaly.explanation)
    if anomaly.recommended_actions:
        pieces.append("**Acoes recomendadas:**")
        for action in anomaly.recommended_actions:
            pieces.append(f"- {action}")
    return "\n\n".join(pieces)


def empty_state(message: str = "Aguardando entrada.", hint: str | None = None) -> str:
    """Bloco HTML informativo para placeholders no estado inicial das abas.

    Args:
        message: titulo curto explicando o estado.
        hint: linha auxiliar opcional com call-to-action (ex.: "Faca upload...").
    """
    safe_msg = html.escape(message)
    if hint:
        safe_hint = html.escape(hint)
        return (
            '<div class="empty-state">'
            f'<div class="empty-state-title">{safe_msg}</div>'
            f'<div class="empty-state-hint">{safe_hint}</div>'
            "</div>"
        )
    return f'<div class="empty-state">{safe_msg}</div>'


def llm_provider_label() -> str:
    """Descreve o provider de LLM ativo, baseado no `.env` atual."""
    from src.config.settings import settings

    if settings.azure_openai_key.get_secret_value() and settings.azure_openai_endpoint:
        return f"Azure OpenAI {settings.azure_openai_deployment} (via AI Foundry)"
    return "Fallback deterministico em markdown (LLM nao configurado)"


def transcription_provider_label() -> str:
    """Descreve o transcritor ativo, baseado em `USE_CLOUD_TRANSCRIPTION`."""
    from src.config.settings import settings

    if settings.use_cloud_transcription and settings.azure_speech_key.get_secret_value():
        return "Azure Speech (PT-BR)"
    return "faster-whisper local (small)"


def emotion_provider_label() -> str:
    """Descreve o classificador de emocao facial ativo.

    Quando `AZURE_OPENAI_VISION_DEPLOYMENT` esta preenchido e o cliente Azure
    consegue inicializar, o pipeline usa GPT-vision multimodal. Caso
    contrario, cai no FER local (com vies conhecido de FER-2013).
    """
    from src.config.settings import settings

    if (
        settings.azure_openai_vision_deployment
        and settings.azure_openai_key.get_secret_value()
        and settings.azure_openai_endpoint
    ):
        return f"Azure OpenAI vision ({settings.azure_openai_vision_deployment})"
    return "FER local"


def vocal_emotion_provider_label() -> str:
    """Descreve o classificador de emocao vocal ativo.

    Quando `AZURE_OPENAI_AUDIO_DEPLOYMENT` esta preenchido e o cliente Azure
    consegue inicializar, o pipeline usa GPT-4o multimodal. Caso contrario,
    cai no wav2vec2 local (com vies conhecido em PT-BR; ver relatorio).
    """
    from src.config.settings import settings

    if (
        settings.azure_openai_audio_deployment
        and settings.azure_openai_key.get_secret_value()
        and settings.azure_openai_endpoint
    ):
        return f"Azure OpenAI multimodal ({settings.azure_openai_audio_deployment})"
    return "wav2vec2 local (superb-er)"


def sentiment_provider_label() -> str:
    """Descreve o provider de sentimento textual."""
    from src.config.settings import settings

    if settings.azure_language_key.get_secret_value() and settings.azure_language_endpoint:
        return "Azure Language"
    return "Nao configurado"


def section_title(title: str, subtitle: str | None = None) -> str:
    """Titulo de secao com barra colorida lateral (consome `.section-title` do tema).

    Args:
        title: titulo curto, peso 600.
        subtitle: linha auxiliar opcional logo abaixo.
    """
    safe_title = html.escape(title)
    out = f'<div class="section-title">{safe_title}</div>'
    if subtitle:
        out += f'<div class="section-subtitle">{html.escape(subtitle)}</div>'
    return out


def kpi_tile(label: str, value: str, hint: str | None = None) -> str:
    """Tile de KPI usado nos cabecalhos de resultado (consome `.kpi-tile` do tema).

    Args:
        label: rotulo em caixa-alta (vira `text-transform: uppercase` no CSS).
        value: valor principal em destaque.
        hint: linha auxiliar opcional.
    """
    hint_html = f'<div class="kpi-hint">{html.escape(hint)}</div>' if hint else ""
    return (
        '<div class="kpi-tile">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f"{hint_html}"
        "</div>"
    )


def kpi_grid(tiles: list[str]) -> str:
    """Envolve uma lista de `kpi_tile` em um grid responsivo."""
    return '<div class="kpi-grid">' + "".join(tiles) + "</div>"


def progress_breadcrumb(active_step: int, steps: list[str] | None = None) -> str:
    """Breadcrumb de progresso (Upload -> Processar -> Resultado).

    Args:
        active_step: indice 1-based da etapa em destaque.
        steps: rotulos das etapas; default `["Upload", "Processar", "Resultado"]`.
    """
    rotulos = steps or ["Upload", "Processar", "Resultado"]
    items: list[str] = []
    for i, rotulo in enumerate(rotulos, start=1):
        active = "active" if i == active_step else ""
        items.append(
            f'<span class="step {active}"><span class="num">{i}</span>{html.escape(rotulo)}</span>'
        )
        if i < len(rotulos):
            items.append('<span class="sep">&rsaquo;</span>')
    return '<div class="breadcrumb">' + "".join(items) + "</div>"


def footer_html(version: str = "0.1.0", repo_url: str | None = None) -> str:
    """Rodape do app com versao e link opcional para o repositorio."""
    repo_part = (
        f'<a href="{html.escape(repo_url)}" target="_blank" rel="noopener">repositorio</a>'
        if repo_url
        else ""
    )
    pieces: list[str] = [f"medica-ia-multimodal v{html.escape(version)}"]
    if repo_part:
        pieces.append(repo_part)
    pieces.append("MIT License")
    body = " &middot; ".join(pieces)
    return f'<div class="app-footer"><div>{body}</div></div>'


def build_acoustic_plot(features: AcousticFeatures | None):
    """Gera figura matplotlib resumindo as features acusticas em barras.

    Args:
        features: saida do pipeline de audio.

    Returns:
        `matplotlib.figure.Figure` ou `None` quando features estao indisponiveis.
        Devolvendo `None`, o `gr.Plot` mostra estado vazio sem erro.
    """
    if features is None:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metricas = [
        ("Pitch medio (Hz)", features.pitch_mean_hz),
        ("Pitch std (Hz)", features.pitch_std_hz),
        ("RMS x100", features.energy_rms * 100),
        ("ZCR x100", features.zero_crossing_rate * 100),
        ("Jitter x100", features.jitter * 100),
        ("Shimmer x100", features.shimmer * 100),
    ]
    labels = [m[0] for m in metricas]
    valores = [m[1] for m in metricas]

    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=BG_SURFACE)
    ax.set_facecolor(BG_SURFACE)
    bars = ax.barh(labels, valores, color=PRIMARY, edgecolor=BORDER, height=0.6)
    ax.invert_yaxis()
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, valor in zip(bars, valores, strict=False):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {valor:.2f}",
            va="center",
            ha="left",
            color=TEXT_PRIMARY,
            fontsize=9,
        )
    ax.set_xlim(left=0)
    ax.set_title(
        "Features acusticas",
        color=TEXT_PRIMARY,
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    fig.tight_layout()
    return fig


def build_video_timeline_plot(events: list[VideoEvent] | None):
    """Gera figura matplotlib com timeline de eventos do video.

    Cada frame amostrado vira uma marca no eixo do tempo; frames com
    deteccoes ganham cor `PRIMARY`, frames sem deteccoes ficam discretos.

    Args:
        events: lista vinda do pipeline de video.

    Returns:
        `matplotlib.figure.Figure` ou `None` quando nao ha eventos.
    """
    if not events:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps_s = [e.timestamp_ms / 1000.0 for e in events]
    detections_count = [len(e.detections) for e in events]
    cores = [PRIMARY if c > 0 else TEXT_TERTIARY for c in detections_count]

    fig, ax = plt.subplots(figsize=(6.5, 2.4), facecolor=BG_SURFACE)
    ax.set_facecolor(BG_SURFACE)
    ax.scatter(
        timestamps_s,
        [1] * len(timestamps_s),
        s=[max(40, c * 60 + 40) for c in detections_count],
        c=cores,
        edgecolors=BORDER,
        linewidths=0.6,
    )
    ax.set_yticks([])
    ax.set_xlabel("Tempo (s)", color=TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    total_det = sum(detections_count)
    titulo = f"Timeline de eventos ({len(events)} frames, {total_det} deteccoes)"
    ax.set_title(titulo, color=TEXT_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=8)
    fig.tight_layout()
    return fig
