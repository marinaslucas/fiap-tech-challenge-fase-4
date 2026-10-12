"""Testes smoke para o modulo de configuracao."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from src.config.settings import PROJECT_ROOT, Settings, settings


@pytest.mark.smoke
def test_settings_singleton_e_uma_instancia_de_settings():
    """A instancia exportada deve ser do tipo Settings."""
    assert isinstance(settings, Settings)


@pytest.mark.smoke
def test_settings_carrega_defaults_quando_env_ausente():
    """Sem env vars definidas, defaults conservadores devem prevalecer."""
    fresh = Settings(_env_file=None)  # ignora .env do disco
    assert fresh.use_cloud_transcription is False
    assert fresh.azure_openai_deployment == "gpt-4.1-mini"
    assert fresh.log_level == "INFO"


@pytest.mark.smoke
def test_chaves_sensiveis_usam_secretstr():
    """Chaves Azure sao SecretStr para evitar log acidental."""
    assert isinstance(settings.azure_speech_key, SecretStr)
    assert isinstance(settings.azure_openai_key, SecretStr)


@pytest.mark.smoke
def test_caminhos_absolutos_sao_resolvidos_contra_project_root():
    """Os helpers devem ancorar paths relativos na raiz do projeto."""
    assert settings.yolo_weights_absolute().is_absolute()
    assert settings.rag_index_absolute().is_absolute()
    assert settings.yolo_weights_absolute().is_relative_to(PROJECT_ROOT)


@pytest.mark.smoke
def test_project_root_aponta_para_raiz_do_repo():
    """PROJECT_ROOT deve ser o diretorio que contem pyproject.toml."""
    assert isinstance(PROJECT_ROOT, Path)
    assert (PROJECT_ROOT / "pyproject.toml").exists()
