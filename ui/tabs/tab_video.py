"""Aba "Vıdeo" da UI Gradio.

Permite upload de um vıdeo, dispara o `VideoPipeline` e mostra:

- Resumo (badge de risco + tipo de cena detectado + fonte do classifier de emocao)
- KPIs (frames, deteccoes, emocoes, linguagem corporal, classes)
- Miniatura do frame com bboxes desenhadas (evidencia visual do que o YOLO viu)
- Timeline de eventos
- Eventos por janela em tabela
- JSON bruto colapsado

Recebe o `VideoPipeline` direto (em vez de so um callable) pra acessar:
- `pipeline.process(path)` para os eventos
- `pipeline.last_scene_type` para o badge de cena
- `pipeline.emotion_classifier` para identificar fonte (Azure ou FER local)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from ui.components import (
    VIDEO_WINDOWS_HEADERS,
    build_detection_thumbnails,
    build_video_timeline_plot,
    emotion_source_label,
    empty_state,
    kpi_grid,
    kpi_tile,
    progress_breadcrumb,
    risk_badge,
    scene_type_badge,
    section_title,
    video_events_to_windowed_rows,
)
from ui.limits import validate_video

if TYPE_CHECKING:
    from src.video.pipeline import VideoPipeline

logger = logging.getLogger(__name__)


def render(video_pipeline: VideoPipeline) -> None:
    """Constroi a aba de vıdeo dentro do contexto atual de `gr.Blocks`.

    Args:
        video_pipeline: instancia compartilhada de `VideoPipeline` (do
            orquestrador). Permite acesso a metadata pos-execucao
            (last_scene_type, emotion_classifier) para enriquecer o Resumo.
    """
    gr.HTML(progress_breadcrumb(1))

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes="input-col"):
            with gr.Group(elem_classes="input-card"):
                gr.HTML(
                    section_title(
                        "Entrada",
                        "Faca upload de um vıdeo curto (mp4/mov, recomendado <60s) "
                        "para extrair eventos por frame: deteccoes YOLO e analise "
                        "multimodal de emocao + linguagem corporal via GPT-vision.",
                    )
                )
                video_input = gr.Video(label="Vıdeo de entrada", sources=["upload"])
                analyze_btn = gr.Button("Analisar vıdeo", variant="primary")

        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML(section_title("Resumo"))
                status_html = gr.HTML(
                    value=empty_state(
                        "Nenhum vıdeo analisado ainda.",
                        hint="Faca upload e clique em Analisar para gerar o resumo.",
                    )
                )
                kpis_html = gr.HTML(value="")

    detection_section = gr.Column(visible=False)
    with detection_section:
        gr.HTML(
            section_title(
                "Frames com deteccoes",
                "Top 4 frames com mais deteccoes, ordenados temporalmente. "
                "Bboxes coloridas por classe (grasper=verde, l_hook=magenta, "
                "blood=vermelho).",
            )
        )
        detection_thumb = gr.Gallery(
            label="",
            show_label=False,
            columns=4,
            rows=1,
            height="auto",  # ajusta natural à proporção dos frames (sem scroll)
            object_fit="contain",
            allow_preview=False,
            show_share_button=False,
            show_download_button=False,
            elem_classes="compact-gallery",
        )

    with gr.Group():
        gr.HTML(section_title("Timeline de eventos"))
        timeline_plot = gr.Plot(label="", show_label=False)

    with gr.Group():
        gr.HTML(
            section_title(
                "Eventos por janela",
                "Resumo agregado em janelas de 5s: deteccoes acumuladas, classes "
                "com confianca media, emocao e linguagem corporal predominantes. "
                "JSON bruto abaixo tem os eventos cru por frame.",
            )
        )
        events_table = gr.Dataframe(
            headers=VIDEO_WINDOWS_HEADERS,
            datatype=["str"] * len(VIDEO_WINDOWS_HEADERS),
            wrap=True,
            interactive=False,
            value=[],
            elem_classes="compact-table",  # colapsa espaco "pra crescer" do Gradio
        )

    with gr.Accordion("JSON bruto (50 primeiros eventos)", open=False):
        raw_json = gr.JSON(value={"events": []})

    def _on_analyze(
        video_path: str | None,
        progress=gr.Progress(),  # noqa: B008  # pattern idiomatico do Gradio
    ):
        empty_result = (
            empty_state(
                "Faca upload de um vıdeo antes de analisar.",
                hint="O arquivo precisa ser mp4 ou mov.",
            ),
            "",
            None,
            None,
            None,
            [],
            {"events": []},
        )
        if not video_path:
            return empty_result

        validation = validate_video(Path(video_path))
        if not validation.ok:
            logger.warning("Video rejeitado na validacao: %s", validation.message)
            return (
                empty_state("Video fora dos limites aceitos.", hint=validation.message),
                "",
                None,
                None,
                None,
                [],
                {"events": []},
            )

        # Adapter: gr.Progress espera `progress(frac, desc=...)`;
        # VideoPipeline.process chama `cb(frac, desc)` posicional.
        def _progress_cb(frac: float, desc: str) -> None:
            progress(frac, desc=desc)

        try:
            events = video_pipeline.process(
                Path(video_path), progress=_progress_cb,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            logger.warning("Falha ao processar vıdeo na UI: %s", exc)
            return (
                empty_state("Falha ao processar vıdeo.", hint=str(exc)),
                "",
                None,
                None,
                None,
                [],
                {"events": []},
            )

        # --- Metricas brutas ---
        from collections import Counter

        total_detections = sum(len(e.detections) for e in events)
        emotions_with_face = sum(1 for e in events if e.facial_emotion is not None)
        classes_set = {d.class_name for e in events for d in e.detections}

        # Linguagem corporal predominante (vinda do GPT-vision via EmotionScore)
        body_labels = [
            e.facial_emotion.body_language for e in events
            if e.facial_emotion is not None and e.facial_emotion.body_language
        ]
        body_counts: Counter[str] = Counter(body_labels)
        # Remove "indefinida" pra preferir categoria interpretavel
        valid_body = {k: v for k, v in body_counts.items() if k != "indefinida"}
        dominant_body = (
            max(valid_body, key=valid_body.get) if valid_body else None
        )
        body_frames = sum(body_counts.values())

        # --- Estado da cena + decisoes do pipeline ---
        scene_type = getattr(video_pipeline, "last_scene_type", None)
        scene_value = scene_type.value if scene_type else "desconhecido"
        scene_is_consultation = scene_value == "consulta"
        scene_is_surgery = scene_value == "cirurgia"

        # --- Fonte da emocao (Azure GPT-vision ou FER local) ---
        emotion_classifier = getattr(video_pipeline, "emotion_classifier", None)
        emotion_src_name = (
            type(emotion_classifier).__name__ if emotion_classifier else "indefinido"
        )
        emotion_src_pretty = emotion_source_label(emotion_src_name)

        # --- Nivel de risco heuristico da aba (so critical se blood detectado) ---
        level = "critical" if "bleeding" in classes_set or "blood" in classes_set else "normal"

        # --- Status block ---
        status_block = (
            '<div style="display: flex; flex-direction: column; gap: 10px;">'
            '<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">'
            f"{risk_badge(level)}"
            f"{scene_type_badge(scene_value)}"
            f'<span style="color: var(--body-text-color); font-size: 12px; opacity: 0.85;">'
            f"Emocao via: <strong>{emotion_src_pretty}</strong>"
            "</span>"
            "</div>"
            '<span style="color: var(--body-text-color); font-size: 12px; opacity: 0.7;">'
            "Heuristica da aba; a classificacao final esta em Multimodal."
            "</span>"
            "</div>"
        )

        # --- KPIs adaptados com hints especificos por cena ---
        if scene_is_consultation:
            detections_value = "0"
            detections_hint = "n/a: detector laparoscopico nao aplica"
            classes_value = "0"
            classes_hint = "sem instrumentos cirurgicos esperados"
        else:
            detections_value = str(total_detections)
            detections_hint = f"{total_detections} bbox em {len(events)} frames"
            classes_value = str(len(classes_set))
            classes_hint = (
                ", ".join(sorted(classes_set))
                if classes_set else "nenhuma classe acima do threshold"
            )

        if scene_is_surgery:
            emotions_value = "n/a"
            emotions_hint = "rosto nao visivel em campo cirurgico"
        else:
            emotions_value = str(emotions_with_face)
            emotions_hint = f"{emotions_with_face} de {len(events)} frames com face"

        if scene_is_surgery:
            body_value = "n/a"
            body_hint = "paciente nao visivel em campo cirurgico"
        elif dominant_body is not None:
            body_value = dominant_body
            body_hint = (
                f"{body_frames} de {len(events)} frames classificados; "
                f"categoria predominante via GPT-vision"
            )
        elif body_frames > 0:
            body_value = "indefinida"
            body_hint = (
                f"{body_frames} de {len(events)} frames com sinal, "
                f"sem categoria interpretavel"
            )
        else:
            body_value = "0"
            body_hint = f"0 de {len(events)} frames com linguagem corporal classificada"

        kpis = kpi_grid(
            [
                kpi_tile("Frames", str(len(events)), hint="amostrados pelo pipeline"),
                kpi_tile("Deteccoes", detections_value, hint=detections_hint),
                kpi_tile("Emocoes", emotions_value, hint=emotions_hint),
                kpi_tile("Linguagem corporal", body_value, hint=body_hint),
                kpi_tile("Classes", classes_value, hint=classes_hint),
            ]
        )

        # --- Miniaturas com bboxes (lista vazia se nao houver deteccao) ---
        thumbs = build_detection_thumbnails(video_path, events, max_thumbs=4)
        has_thumbs = bool(thumbs)

        rows = video_events_to_windowed_rows(events, window_seconds=5.0)
        payload = {"events": [e.model_dump() for e in events[:50]]}
        timeline = build_video_timeline_plot(events)
        return (
            status_block,
            kpis,
            gr.update(value=thumbs if has_thumbs else None),
            gr.update(visible=has_thumbs),
            timeline,
            rows,
            payload,
        )

    analyze_btn.click(
        fn=lambda: gr.update(interactive=False, value="Analisando..."),
        outputs=analyze_btn,
        queue=False,
    ).then(
        fn=_on_analyze,
        inputs=[video_input],
        outputs=[
            status_html,
            kpis_html,
            detection_thumb,
            detection_section,
            timeline_plot,
            events_table,
            raw_json,
        ],
        # show_progress="minimal" pra deixar gr.Progress do callback aparecer
        # com texto detalhado em vez do spinner generico do "full"
        show_progress="minimal",
    ).then(
        fn=lambda: gr.update(interactive=True, value="Analisar vıdeo"),
        outputs=analyze_btn,
        queue=False,
    )

    def _on_clear():
        """Reseta todos os outputs pro estado inicial quando o video e removido."""
        return (
            empty_state(
                "Nenhum vıdeo analisado ainda.",
                hint="Faca upload e clique em Analisar para gerar o resumo.",
            ),
            "",
            gr.update(value=None),
            gr.update(visible=False),
            None,
            [],
            {"events": []},
        )

    video_input.clear(
        fn=_on_clear,
        outputs=[
            status_html,
            kpis_html,
            detection_thumb,
            detection_section,
            timeline_plot,
            events_table,
            raw_json,
        ],
        queue=False,
    )
