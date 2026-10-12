"""Cliente Azure OpenAI multimodal (audio) para classificacao de emocao vocal.

Envia WAV em base64 + prompt JSON-mode pra um deployment de modelo de audio
(ex: `gpt-4o-mini-audio-preview`) no mesmo endpoint do Foundry usado pelo
`AzureOpenAIClient`. Retorna `EmotionScore` ou `None` se o servico nao
estiver configurado/disponivel.

Motivacao: o `wav2vec2-base-superb-er` foi treinado em RAVDESS (atores
americanos, fala teatral) e atribui `angry` com confianca >0.95 a
praticamente qualquer voz feminina em PT-BR. Um LLM multimodal analisa o
sinal acustico junto com o conteudo da fala, sem esse vies de dominio.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from src.audio.types import EmotionScore
from src.config.settings import settings

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

SYSTEM_PROMPT: str = (
    "Voce e um analista clinico especializado em saude da mulher. Analise "
    "o audio fornecido e classifique o estado emocional predominante da "
    "pessoa que esta falando, considerando tanto pistas prosodicas (tom, "
    "ritmo, energia, hesitacoes) quanto o conteudo semantico da fala em "
    "PT-BR. Responda EXCLUSIVAMENTE em JSON valido no formato definido "
    "no prompt do usuario."
)


def _normalize_base_url(endpoint: str) -> str:
    """Normaliza endpoint do Foundry (mesma logica de azure_openai.py)."""
    base = endpoint.rstrip("/")
    for suffix in ("/responses", "/chat/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


class AzureOpenAIAudioEmotion:
    """Cliente fino sobre `openai` SDK que classifica emocao vocal via GPT-4o-audio.

    Mesma assinatura que `VocalEmotionClassifier.classify` para permitir
    substituicao plug-and-play no pipeline de audio. Em caso de falha,
    retorna `None` (chamador degrada para wav2vec2 local ou pula o pilar).
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
        self.deployment: str = settings.azure_openai_audio_deployment
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.timeout_s: float = timeout_s
        self._client = None
        self._available: bool = False
        self._load_attempted: bool = False

    @property
    def is_configured(self) -> bool:
        """`True` se chave, endpoint e deployment de audio estao definidos."""
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
                "Azure OpenAI audio cliente inicializado (deployment=%s, base_url=%s)",
                self.deployment,
                base_url,
            )
        except (ImportError, ValueError) as exc:
            logger.warning("Falha ao iniciar Azure OpenAI audio: %s", exc)
            self._available = False

    def classify(self, audio_path: Path) -> EmotionScore | None:
        """Classifica a emocao predominante do audio.

        Args:
            audio_path: caminho do WAV (preferencia 16 kHz mono).

        Returns:
            `EmotionScore` com label, confidence e dict scores (apenas a
            classe predita), ou `None` se o servico nao estiver disponivel
            ou se a resposta vier malformada.
        """
        self._ensure_client()
        if not self._available or self._client is None:
            return None

        if not audio_path.exists():
            logger.warning("Audio nao encontrado para classificacao: %s", audio_path)
            return None

        try:
            audio_bytes = audio_path.read_bytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        except OSError as exc:
            logger.warning("Falha ao ler %s: %s", audio_path, exc)
            return None

        user_prompt = (
            "Classifique a emocao predominante na fala. Responda APENAS com "
            "o seguinte JSON (sem comentarios, sem markdown):\n"
            "{\n"
            '  "label": "<uma de: ' + ", ".join(ALLOWED_LABELS) + '>",\n'
            '  "confidence": <float entre 0.0 e 1.0>,\n'
            '  "reasoning": "<uma frase descrevendo as pistas que levaram a classificacao>"\n'
            "}"
        )

        try:
            # NAO usar response_format={'type': 'json_object'}: o gpt-audio-mini
            # do Foundry retorna 400 com esse parametro. O prompt ja instrui
            # JSON puro e _parse_json_response cuida de markdown fences se vierem.
            response = self._client.chat.completions.create(
                model=self.deployment,
                modalities=["text"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - SDK levanta varias subclasses
            logger.warning("Azure OpenAI audio falhou: %s", exc)
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
        logger.warning("Resposta nao-JSON do LLM de audio: %s | content=%r", exc, content[:120])
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

    return EmotionScore(
        label=label_raw,
        confidence=confidence,
        scores={label_raw: confidence},
    )
