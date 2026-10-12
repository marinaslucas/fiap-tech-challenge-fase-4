"""Testes smoke para a camada de UI (tema Gradio + CSS customizado + helpers)."""

import gradio as gr
import pytest

from src.audio.types import AcousticFeatures
from src.video.types import BoundingBox, Detection, VideoEvent
from ui import theme
from ui.components import (
    build_acoustic_plot,
    build_video_timeline_plot,
    empty_state,
    footer_html,
    header_html,
    kpi_grid,
    kpi_tile,
    progress_breadcrumb,
    risk_badge,
    section_title,
)
from ui.theme import get_custom_css, get_theme


@pytest.mark.smoke
def test_get_theme_retorna_subclasse_de_themeclass():
    """O tema customizado deve ser uma `ThemeClass` valida para `gr.Blocks`."""
    built = get_theme()
    assert isinstance(built, gr.themes.ThemeClass)


@pytest.mark.smoke
def test_get_theme_aplica_fonte_inter():
    """A fonte primaria deve ser Inter conforme docs/arquitetura/design.md."""
    built = get_theme()
    # A fonte e armazenada como tupla de GradioFont; basta checar a primeira entrada
    primary_font = built._font[0] if isinstance(built._font, (list, tuple)) else built._font
    assert "Inter" in str(primary_font)


@pytest.mark.smoke
def test_get_custom_css_contem_tokens_da_paleta():
    """O CSS deve referenciar os hexa principais do design system."""
    css = get_custom_css()
    assert theme.PRIMARY in css
    assert theme.TEXT_PRIMARY in css
    assert theme.ALERT_NORMAL_BG in css
    assert theme.ALERT_MODERATE_BG in css
    assert theme.ALERT_CRITICAL_BG in css


@pytest.mark.smoke
def test_get_custom_css_declara_classes_de_alerta():
    """As tres classes de alerta e o logo precisam existir para os componentes."""
    css = get_custom_css()
    for selector in (".logo", ".alert-normal", ".alert-moderate", ".alert-critical"):
        assert selector in css


@pytest.mark.smoke
def test_get_custom_css_importa_fonte_inter_via_google_fonts():
    """O @import garante a fonte mesmo quando self-hosted nao estiver disponivel."""
    css = get_custom_css()
    assert "fonts.googleapis.com" in css
    assert "family=Inter" in css


@pytest.mark.smoke
def test_tokens_de_cor_sao_hex_strings_validas():
    """Tokens exportados devem ser hex strings de 7 caracteres (`#RRGGBB`)."""
    tokens = [
        theme.BG_PRIMARY,
        theme.BG_SURFACE,
        theme.BG_ELEVATED,
        theme.BORDER,
        theme.PRIMARY,
        theme.PRIMARY_HOVER,
        theme.PRIMARY_SUBTLE,
        theme.TEXT_PRIMARY,
        theme.TEXT_SECONDARY,
        theme.TEXT_TERTIARY,
        theme.ALERT_NORMAL_BG,
        theme.ALERT_NORMAL_FG,
        theme.ALERT_MODERATE_BG,
        theme.ALERT_MODERATE_FG,
        theme.ALERT_CRITICAL_BG,
        theme.ALERT_CRITICAL_FG,
    ]
    for token in tokens:
        assert isinstance(token, str)
        assert token.startswith("#")
        assert len(token) == 7


@pytest.mark.smoke
def test_header_html_usa_app_header_e_status_pill():
    """O header novo deve consumir as classes `.app-header` e `.status-pill`."""
    out = header_html("Sistema online")
    assert 'class="app-header"' in out
    assert 'class="status-pill"' in out
    assert "medica-" in out and "multimodal" in out


@pytest.mark.smoke
def test_kpi_tile_e_kpi_grid_geram_html_estruturado():
    """`kpi_tile` deve incluir label/value/hint e `kpi_grid` envolver em grid."""
    tile = kpi_tile("Triggers", "3", hint="regras")
    assert "kpi-tile" in tile
    assert "Triggers" in tile and "3" in tile and "regras" in tile

    grid = kpi_grid([tile, kpi_tile("Modalidades", "2")])
    assert grid.startswith('<div class="kpi-grid">')
    assert grid.count("kpi-tile") == 2


@pytest.mark.smoke
def test_progress_breadcrumb_marca_step_ativo():
    """A etapa indicada deve receber a classe `active`; as outras nao."""
    crumb = progress_breadcrumb(2)
    assert 'class="step active"' in crumb
    assert crumb.count("step active") == 1
    assert "Upload" in crumb and "Processar" in crumb and "Resultado" in crumb


@pytest.mark.smoke
def test_section_title_aceita_subtitulo():
    """`section_title` deve renderizar subtitulo opcional como secao-subtitulo."""
    com_sub = section_title("Resumo", "Detalhes do caso")
    assert 'class="section-title"' in com_sub
    assert 'class="section-subtitle"' in com_sub
    assert "Detalhes do caso" in com_sub

    sem_sub = section_title("Resumo")
    assert "section-subtitle" not in sem_sub


@pytest.mark.smoke
def test_empty_state_aceita_hint():
    """`empty_state` com hint deve renderizar titulo + linha auxiliar."""
    out = empty_state("Sem dados", hint="Faca upload primeiro.")
    assert "empty-state-title" in out and "empty-state-hint" in out
    assert "Sem dados" in out and "Faca upload primeiro." in out


@pytest.mark.smoke
def test_footer_html_inclui_versao_e_link():
    """O footer deve mostrar versao e, se houver, link clicavel."""
    out = footer_html(version="1.2.3", repo_url="https://example.test")
    assert "app-footer" in out
    assert "1.2.3" in out
    assert 'href="https://example.test"' in out


@pytest.mark.smoke
def test_risk_badge_usa_classes_alert():
    """`risk_badge` deve usar as 3 classes CSS do tema, uma por nivel."""
    assert "alert-normal" in risk_badge("normal")
    assert "alert-moderate" in risk_badge("moderate")
    assert "alert-critical" in risk_badge("critical")


@pytest.mark.smoke
def test_build_acoustic_plot_retorna_figure_quando_features_existem():
    """O plot deve produzir uma `matplotlib.figure.Figure` ou `None` se vazio."""
    feats = AcousticFeatures(
        duration_s=2.0,
        pitch_mean_hz=200.0,
        pitch_std_hz=10.0,
        energy_rms=0.05,
        zero_crossing_rate=0.1,
        jitter=0.01,
        shimmer=0.02,
        mfcc_means=[0.0] * 13,
    )
    import matplotlib.figure

    fig = build_acoustic_plot(feats)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert build_acoustic_plot(None) is None


@pytest.mark.smoke
def test_build_video_timeline_plot_retorna_figure_quando_ha_eventos():
    """O timeline deve produzir uma figura ou `None` para lista vazia."""
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    events = [
        VideoEvent(
            frame_index=i,
            timestamp_ms=i * 1000,
            detections=[Detection(class_id=0, class_name="grasper", confidence=0.9, bbox=bbox)]
            if i % 2 == 0
            else [],
        )
        for i in range(4)
    ]
    import matplotlib.figure

    fig = build_video_timeline_plot(events)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert build_video_timeline_plot([]) is None
    assert build_video_timeline_plot(None) is None


@pytest.mark.smoke
def test_css_customizado_contem_novos_blocos():
    """O CSS deve declarar os blocos visuais que os componentes consomem."""
    css = get_custom_css()
    for selector in (
        ".app-header",
        ".status-pill",
        ".kpi-tile",
        ".kpi-grid",
        ".section-title",
        ".breadcrumb",
        ".empty-state",
        ".app-footer",
    ):
        assert selector in css, f"selector {selector!r} ausente no CSS"
