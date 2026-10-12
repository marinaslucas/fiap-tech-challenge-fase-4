"""Cliente Azure Language: sentimento + key phrases sobre transcricao textual.

Sem credenciais, retorna `None`/lista vazia e o pipeline continua funcional.
"""

from __future__ import annotations

import logging

from src.audio.types import SentimentResult
from src.config.settings import settings

logger = logging.getLogger(__name__)


class AzureLanguageClient:
    """Cliente do Azure Language Service para sentimento e key phrases."""

    def __init__(self, language: str = "pt") -> None:
        """Configura o cliente.

        Args:
            language: codigo ISO 639-1 do idioma da analise (default `pt`).
        """
        self.language: str = language
        self.api_key: str = settings.azure_language_key.get_secret_value()
        self.endpoint: str = settings.azure_language_endpoint
        self._client = None
        self._available: bool = False
        self._load_attempted: bool = False

    @property
    def is_configured(self) -> bool:
        """`True` se a chave e o endpoint do Azure Language estao definidos."""
        return bool(self.api_key and self.endpoint)

    def _ensure_client(self) -> None:
        """Lazy-load do cliente do SDK Azure."""
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.is_configured:
            self._available = False
            return
        try:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential

            self._client = TextAnalyticsClient(
                endpoint=self.endpoint, credential=AzureKeyCredential(self.api_key)
            )
            self._available = True
            logger.info("Azure Language cliente inicializado")
        except (ImportError, ValueError) as exc:
            logger.warning("Falha ao iniciar Azure Language: %s", exc)
            self._available = False

    def analyze_sentiment(self, text: str) -> SentimentResult | None:
        """Analisa sentimento de um texto.

        Returns:
            `SentimentResult` com label dominante e distribuicao,
            ou `None` se o servico nao estiver configurado/disponivel.
        """
        if not text.strip():
            return None
        self._ensure_client()
        if not self._available or self._client is None:
            return None
        response = self._client.analyze_sentiment(documents=[text], language=self.language)
        result = response[0]
        if getattr(result, "is_error", False):
            logger.warning("Azure Sentiment erro: %s", getattr(result, "error", None))
            return None
        scores = {
            "positive": float(result.confidence_scores.positive),
            "neutral": float(result.confidence_scores.neutral),
            "negative": float(result.confidence_scores.negative),
        }
        label = result.sentiment
        # Azure usa "mixed" como meta-label quando o texto tem partes
        # positivas e negativas, mas nao retorna confidence pra ele.
        # Pra esse caso, expomos a maior das polaridades (a que domina
        # marginalmente o conflito), evitando KPI sempre 0%.
        if label == "mixed":
            confidence = max(scores["positive"], scores["negative"])
        else:
            confidence = scores.get(label, 0.0)
        return SentimentResult(label=label, confidence=confidence, scores=scores)

    def extract_key_phrases(self, text: str) -> list[str]:
        """Extrai frases-chave de um texto.

        Returns:
            Lista de strings (vazia se servico indisponivel ou texto vazio).
        """
        if not text.strip():
            return []
        self._ensure_client()
        if not self._available or self._client is None:
            return []
        response = self._client.extract_key_phrases(documents=[text], language=self.language)
        result = response[0]
        if getattr(result, "is_error", False):
            logger.warning("Azure Key Phrases erro: %s", getattr(result, "error", None))
            return []
        return list(result.key_phrases)
