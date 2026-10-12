"""Testes smoke do cliente Azure OpenAI.

Sem chamadas reais ao servico: mocks isolam o SDK `openai` e validam
`is_configured` + retorno `None` quando o servico nao esta configurado.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm import AzureOpenAIClient


@pytest.mark.smoke
def test_client_nao_configurado_retorna_none(monkeypatch):
    # Forca settings sem credenciais
    from pydantic import SecretStr

    from src.config.settings import settings as settings_obj

    monkeypatch.setattr(settings_obj, "azure_openai_key", SecretStr(""))
    monkeypatch.setattr(settings_obj, "azure_openai_endpoint", "")
    client = AzureOpenAIClient()
    assert client.is_configured is False
    assert client.chat([{"role": "user", "content": "ola"}]) is None


@pytest.mark.smoke
def test_client_configurado_chama_sdk_e_retorna_texto(monkeypatch):
    from pydantic import SecretStr

    from src.config.settings import settings as settings_obj

    monkeypatch.setattr(settings_obj, "azure_openai_key", SecretStr("fake-key"))
    monkeypatch.setattr(settings_obj, "azure_openai_endpoint", "https://example.openai.azure.com/")
    monkeypatch.setattr(settings_obj, "azure_openai_deployment", "gpt-4o-mini")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="resposta gerada"))]
    fake_completions = MagicMock()
    fake_completions.create.return_value = fake_response
    fake_chat = MagicMock(completions=fake_completions)
    fake_client = MagicMock(chat=fake_chat)

    with patch("openai.OpenAI", return_value=fake_client):
        client = AzureOpenAIClient()
        out = client.chat([{"role": "user", "content": "ola"}])

    assert out == "resposta gerada"
    fake_completions.create.assert_called_once()
    kwargs = fake_completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["messages"][0]["content"] == "ola"


@pytest.mark.smoke
def test_client_chat_vazio_retorna_none(monkeypatch):
    from pydantic import SecretStr

    from src.config.settings import settings as settings_obj

    monkeypatch.setattr(settings_obj, "azure_openai_key", SecretStr("fake-key"))
    monkeypatch.setattr(settings_obj, "azure_openai_endpoint", "https://example.openai.azure.com/")
    monkeypatch.setattr(settings_obj, "azure_openai_deployment", "gpt-4o-mini")
    with patch("openai.OpenAI", return_value=MagicMock()):
        client = AzureOpenAIClient()
        assert client.chat([]) is None


@pytest.mark.smoke
def test_client_trata_excecao_do_sdk_e_retorna_none(monkeypatch):
    from pydantic import SecretStr

    from src.config.settings import settings as settings_obj

    monkeypatch.setattr(settings_obj, "azure_openai_key", SecretStr("fake-key"))
    monkeypatch.setattr(settings_obj, "azure_openai_endpoint", "https://example.openai.azure.com/")
    monkeypatch.setattr(settings_obj, "azure_openai_deployment", "gpt-4o-mini")

    fake_completions = MagicMock()
    fake_completions.create.side_effect = RuntimeError("limite atingido")
    fake_client = MagicMock(chat=MagicMock(completions=fake_completions))
    with patch("openai.OpenAI", return_value=fake_client):
        client = AzureOpenAIClient()
        assert client.chat([{"role": "user", "content": "x"}]) is None
