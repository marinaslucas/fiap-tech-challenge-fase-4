"""Aba "Audio" da UI Gradio.

Permite upload de um arquivo de audio, dispara o `AudioPipeline` e exibe
transcricao, features acusticas, emocao vocal e sentimento.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from ui.components import (
    build_acoustic_plot,
    empty_state,
    format_audio_summary,
    kpi_grid,
    kpi_tile,
    progress_breadcrumb,
    risk_badge,
    section_title,
    sentiment_provider_label,
    transcription_provider_label,
    vocal_emotion_provider_label,
)
from ui.limits import validate_audio

if TYPE_CHECKING:
    from src.audio.types import AudioAnalysis

logger = logging.getLogger(__name__)

AudioProcessor = Callable[[Path], "AudioAnalysis"]


def render(process_audio: AudioProcessor) -> None:
    """Constroi a aba de audio dentro do contexto atual de `gr.Blocks`.

    Args:
        process_audio: callable que recebe o path do audio e devolve um
            `AudioAnalysis`. Tipicamente `lambda p: orchestrator
            .audio_pipeline.process(p)`. Permite injetar mocks em testes.
    """
    gr.HTML(progress_breadcrumb(1))

    with gr.Group():
        gr.HTML(
            section_title(
                "Entrada",
                "Faca upload de um audio (wav/mp3) para extrair transcricao, "
                "features acusticas, emocao vocal e sentimento do texto.",
            )
        )
        audio_input = gr.Audio(
            label="Audio de entrada", sources=["upload"], type="filepath"
        )
        analyze_btn = gr.Button("Analisar audio", variant="primary")

    with gr.Group():
        gr.HTML(section_title("Resumo"))
        status_html = gr.HTML(
            value=empty_state(
                "Nenhum audio analisado ainda.",
                hint="Faca upload e clique em Analisar para gerar o resumo.",
            )
        )
        kpis_html = gr.HTML(value="")

    with gr.Group():
        gr.HTML(
            section_title(
                "Transcricao, emocao e sentimento",
                (
                    f"Transcricao: {transcription_provider_label()}. "
                    f"Sentimento: {sentiment_provider_label()}."
                ),
            )
        )
        with gr.Column(elem_classes="content-box"):
            summary_md = gr.Markdown(value="")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML(section_title("Grafico de features"))
                features_plot = gr.Plot(label="", show_label=False)
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML(section_title("Features acusticas"))
                with gr.Column(elem_classes="content-box"):
                    features_md = gr.Markdown(value="")

    with gr.Accordion("JSON bruto da analise", open=False):
        raw_json = gr.JSON(value={})

    def _on_analyze(
        audio_path: str | None,
        progress=gr.Progress(),  # noqa: B008  # pattern idiomatico do Gradio
    ):
        if not audio_path:
            return (
                empty_state(
                    "Faca upload de um audio antes de analisar.",
                    hint="Formatos aceitos: wav, mp3.",
                ),
                "",
                "",
                None,
                "",
                {},
            )

        validation = validate_audio(Path(audio_path))
        if not validation.ok:
            logger.warning("Audio rejeitado na validacao: %s", validation.message)
            return (
                empty_state("Audio fora dos limites aceitos.", hint=validation.message),
                "",
                "",
                None,
                "",
                {},
            )

        # Adapter: gr.Progress espera `progress(frac, desc=...)`;
        # AudioPipeline.process chama `cb(frac, desc)` posicional.
        def _progress_cb(frac: float, desc: str) -> None:
            progress(frac, desc=desc)

        try:
            analysis = process_audio(Path(audio_path), progress=_progress_cb)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("Falha ao processar audio na UI: %s", exc)
            return (
                empty_state("Falha ao processar audio.", hint=str(exc)),
                "",
                "",
                None,
                "",
                {},
            )

        emotion_label_raw = analysis.emotion.label.lower() if analysis.emotion else "n/d"
        emotion_label = analysis.emotion.label_pt if analysis.emotion else "n/d"
        sentiment_label = analysis.sentiment.label if analysis.sentiment else "n/d"

        # Hint do sentimento muda conforme o label. Pra "mixed" mostramos
        # a distribuicao positive/negative que justifica a marcacao;
        # pros demais, mostramos quem oferece o sinal (Azure Language).
        if analysis.sentiment and sentiment_label == "mixed":
            pos = analysis.sentiment.scores.get("positive", 0.0)
            neg = analysis.sentiment.scores.get("negative", 0.0)
            sentiment_hint = f"pos: {pos:.0%} / neg: {neg:.0%}"
        else:
            sentiment_hint = sentiment_provider_label()

        distress_emotions = {"sad", "angry", "fearful", "fear", "ang", "fea"}
        level = (
            "moderate"
            if sentiment_label == "negative" or emotion_label_raw in distress_emotions
            else "normal"
        )
        status_block = (
            '<div style="display: flex; align-items: center; gap: 12px;">'
            f"{risk_badge(level)}"
            '<span style="color: var(--body-text-color); font-size: 13px;">'
            "Heuristica desta aba (ver aba Multimodal para classificacao completa)."
            "</span></div>"
        )
        duration = (
            f"{analysis.acoustic_features.duration_s:.1f}s" if analysis.acoustic_features else "n/d"
        )
        kpis = kpi_grid(
            [
                kpi_tile("Duracao", duration, hint="audio analisado"),
                kpi_tile("Emocao", emotion_label, hint=vocal_emotion_provider_label()),
                kpi_tile("Sentimento", sentiment_label, hint=sentiment_hint),
                kpi_tile(
                    "Segmentos",
                    str(len(analysis.segments)),
                    hint="transcritos",
                ),
            ]
        )
        plot = build_acoustic_plot(analysis.acoustic_features)
        from ui.components import format_acoustic_features

        return (
            status_block,
            kpis,
            format_audio_summary(analysis),
            plot,
            format_acoustic_features(analysis.acoustic_features),
            analysis.model_dump(),
        )

    analyze_btn.click(
        fn=lambda: gr.update(interactive=False, value="Analisando..."),
        outputs=analyze_btn,
        queue=False,
    ).then(
        fn=_on_analyze,
        inputs=[audio_input],
        outputs=[
            status_html,
            kpis_html,
            summary_md,
            features_plot,
            features_md,
            raw_json,
        ],
        # show_progress="minimal" pra deixar gr.Progress do callback aparecer
        # com texto detalhado em vez do spinner generico do "full"
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(interactive=True, value="Analisar audio"),
        outputs=analyze_btn,
        queue=False,
    )

    def _on_clear():
        """Reseta todos os outputs pro estado inicial quando o audio e removido."""
        return (
            empty_state(
                "Nenhum audio analisado ainda.",
                hint="Faca upload e clique em Analisar para gerar o resumo.",
            ),
            "",
            "",
            None,
            "",
            {},
        )

    audio_input.clear(
        fn=_on_clear,
        outputs=[
            status_html,
            kpis_html,
            summary_md,
            features_plot,
            features_md,
            raw_json,
        ],
        queue=False,
    )
