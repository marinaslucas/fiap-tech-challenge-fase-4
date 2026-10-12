"""Cliente Azure AI Foundry (endpoint v1 OpenAI-compativel).

Sem credenciais, `chat()` retorna `None` e o caller degrada para
fallback deterministico em `src/report.py`.

Endpoint precisa terminar em `/openai/v1`. URLs com `/responses` ou
`/chat/completions` no fim sao normalizadas (Foundry mostra varias
variantes na aba "Consume").
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from src.config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_MAX_TOKENS: int = 800
DEFAULT_TIMEOUT_S: float = 30.0

ChatMessage = dict[str, str]


def _normalize_base_url(endpoint: str) -> str:
    """Remove sufixos do Foundry pra deixar `base_url` no formato esperado pelo SDK OpenAI."""
    base = endpoint.rstrip("/")
    for suffix in ("/responses", "/chat/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


class AzureOpenAIClient:
    """Cliente fino sobre o SDK `openai` apontando para o endpoint v1 do Foundry."""

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Configura o cliente.

        Args:
            temperature: criatividade do modelo (mais baixo = mais factual).
            max_tokens: tamanho maximo da resposta.
            timeout_s: timeout HTTP em segundos.
        """
        self.api_key: str = settings.azure_openai_key.get_secret_value()
        self.endpoint: str = settings.azure_openai_endpoint
        self.deployment: str = settings.azure_openai_deployment
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.timeout_s: float = timeout_s
        self._client = None
        self._available: bool = False
        self._load_attempted: bool = False

    @property
    def is_configured(self) -> bool:
        """`True` se chave, endpoint e deployment estao definidos."""
        return bool(self.api_key and self.endpoint and self.deployment)

    def _ensure_client(self) -> None:
        """Lazy-load do `OpenAI` do SDK `openai` apontando para o endpoint v1."""
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.is_configured:
            self._available = False
            return
        try:
            from openai import OpenAI

            base_url = _normalize_base_url(self.endpoint)
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=self.timeout_s,
            )
            self._available = True
            logger.info(
                "Azure AI Foundry cliente inicializado (deployment=%s, base_url=%s)",
                self.deployment,
                base_url,
            )
        except (ImportError, ValueError) as exc:
            logger.warning("Falha ao iniciar Azure AI Foundry: %s", exc)
            self._available = False

    def chat(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Envia uma sequencia de mensagens e devolve o texto da resposta.

        Args:
            messages: sequencia de dicts no formato OpenAI
                (`{"role": "system|user|assistant", "content": "..."}`).
            temperature: override opcional do default.
            max_tokens: override opcional do default.

        Returns:
            String com o texto da primeira choice, ou `None` se o servico
            nao estiver configurado/disponivel ou se a resposta vier vazia.
        """
        self._ensure_client()
        if not self._available or self._client is None:
            return None

        payload = list(messages)
        if not payload:
            logger.warning("chat() chamado com lista de mensagens vazia.")
            return None

        try:
            response = self._client.chat.completions.create(
                model=self.deployment,
                messages=payload,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - SDK levanta varias subclasses
            logger.warning("Azure AI Foundry chat falhou: %s", exc)
            return None

        if not response.choices:
            return None
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if not content:
            return None
        return content.strip()
