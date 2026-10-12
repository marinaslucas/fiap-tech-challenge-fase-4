"""Cliente Azure OpenAI multimodal (visao) para classificacao de estado emocional.

Envia frame em base64 PNG + prompt JSON-mode pra um deployment de visao
(ex: `gpt-4o-mini` ou `gpt-4o`) no mesmo endpoint do Foundry usado pelo
`AzureOpenAIClient`. Retorna `EmotionScore` compativel com
`src.video.types`.

Motivacoes:

1. FER baseado em FER-2013 tem forte vies pra `angry`/`sad` em rostos
   femininos PT-BR em situacao clinica neutra. LLM multimodal analisa em
   contexto, sem herdar esse vies.
2. Consulta clinica em PT-BR (simulacoes pos-2020) frequentemente usa
   mascaras de EPI, ocultando microexpressoes faciais. O prompt instrui o
   LLM a usar linguagem corporal (postura, gestos, posicionamento) como
   sinal primario, com face como sinal complementar quando visivel.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import settings
from src.video.types import EmotionScore

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 200
DEFAULT_TIMEOUT_S: float = 30.0

# Rotulos compativeis com `EMOTION_LABEL_PT` em src/audio/types.py.
ALLOWED_LABELS: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
)

# Categorias de linguagem corporal predominante. Mapeadas pelo prompt;
# o modelo escolhe uma quando consegue interpretar a cena.
ALLOWED_BODY_LANGUAGE: tuple[str, ...] = (
    "tranquila",
    "tensa",
    "retraida",
    "agitada",
    "indefinida",
)

SYSTEM_PROMPT: str = (
    "Analise visual de cena clinica de saude da mulher. Identifique "
    "visualmente a paciente (pessoa em atendimento, geralmente sentada "
    "frente a profissional de saude) e classifique dois sinais: "
    "(a) emocao aparente, considerando postura, gestos e expressao "
    "facial quando visivel; "
    "(b) linguagem corporal predominante, descrevendo a manifestacao "
    "nao-verbal global em uma categoria. Pessoas em repouso clinico "
    "sao tipicamente neutras e tranquilas. Responda apenas em JSON valido."
)


def _normalize_base_url(endpoint: str) -> str:
    """Normaliza endpoint do Foundry (mesma logica de azure_openai.py)."""
    base = endpoint.rstrip("/")
    for suffix in ("/responses", "/chat/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def _encode_image(image: Path | np.ndarray) -> str | None:
    """Converte imagem (Path ou ndarray BGR) em string base64 PNG.

    Retorna `None` em caso de falha de leitura/encoding.
    """
    if isinstance(image, Path):
        if not image.exists():
            logger.warning("Imagem nao encontrada para classificacao: %s", image)
            return None
        frame = cv2.imread(str(image))
        if frame is None:
            logger.warning("Falha ao ler imagem (cv2.imread retornou None): %s", image)
            return None
    else:
        frame = image

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        logger.warning("Frame invalido recebido (tipo=%s)", type(frame).__name__)
        return None

    success, buffer = cv2.imencode(".png", frame)
    if not success:
        logger.warning("cv2.imencode falhou ao serializar frame PNG")
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


class AzureOpenAIVisionEmotion:
    """Cliente fino sobre `openai` SDK que classifica emocao facial via GPT-4o vision.

    Implementa `FacialEmotionClassifierProtocol` definido em
    `src.video.emotion`, permitindo substituicao plug-and-play do FER.
    Em caso de falha, retorna `None` (chamador degrada para FER local
    ou pula o pilar de emocao facial).
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Configura o cliente lendo credenciais de `settings`."""
        self.api_key: str = settings.azure_openai_key.get_secret_value()
        self.endpoint: str = settings.azure_openai_endpoint
        self.deployment: str = settings.azure_openai_vision_deployment
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.timeout_s: float = timeout_s
        self._client = None
        self._available: bool = False
        self._load_attempted: bool = False

    @property
    def is_configured(self) -> bool:
        """`True` se chave, endpoint e deployment de visao estao definidos."""
        return bool(self.api_key and self.endpoint and self.deployment)

    def _ensure_client(self) -> None:
        """Lazy-load do SDK `openai` apontando para o endpoint v1 do Foundry."""
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
                "Azure OpenAI vision cliente inicializado (deployment=%s, base_url=%s)",
                self.deployment,
                base_url,
            )
        except (ImportError, ValueError) as exc:
            logger.warning("Falha ao iniciar Azure OpenAI vision: %s", exc)
            self._available = False

    def classify(self, image: Path | np.ndarray) -> EmotionScore | None:
        """Classifica a emocao predominante da face na imagem.

        Args:
            image: caminho de arquivo de imagem ou frame BGR (numpy ndarray
                no formato OpenCV, HxWx3).

        Returns:
            `EmotionScore` com label, confidence e dict scores (apenas a
            classe predita), ou `None` se o servico nao estiver disponivel
            ou se a resposta vier malformada.
        """
        self._ensure_client()
        if not self._available or self._client is None:
            return None

        image_b64 = _encode_image(image)
        if image_b64 is None:
            return None

        user_prompt = (
            "Avalie a paciente e retorne JSON com: "
            "label (emocao, uma de: " + ", ".join(ALLOWED_LABELS) + "), "
            "confidence (0.0 a 1.0), "
            "body_language (uma de: " + ", ".join(ALLOWED_BODY_LANGUAGE) + "), "
            "reasoning (frase curta sobre postura, gestos ou expressao)."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.deployment,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - SDK levanta varias subclasses
            logger.warning("Azure OpenAI vision falhou: %s", exc)
            return None

        if not response.choices:
            return None

        content = getattr(response.choices[0].message, "content", None)
        if not content:
            return None

        return _parse_json_response(content)


def _parse_json_response(content: str) -> EmotionScore | None:
    """Converte a string JSON retornada pelo LLM em `EmotionScore` tipado."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("Resposta nao-JSON do LLM de visao: %s | content=%r", exc, content[:120])
        return None

    label_raw = str(data.get("label", "")).strip().lower()
    if label_raw not in ALLOWED_LABELS:
        logger.warning("Label fora do dominio retornado pelo LLM: %r", label_raw)
        return None

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    body_raw = str(data.get("body_language", "")).strip().lower()
    body_language = body_raw if body_raw in ALLOWED_BODY_LANGUAGE else None

    return EmotionScore(
        label=label_raw,
        confidence=confidence,
        scores={label_raw: confidence},
        body_language=body_language,
    )
