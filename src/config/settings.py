"""Configuracao centralizada do projeto.

Carrega variaveis de ambiente do arquivo `.env` (ou do ambiente do processo)
usando Pydantic Settings v2. Este modulo expoe a instancia `settings`,
que e o ponto unico de acesso a configuracao em todo o codigo.

Uso:
    from src.config.settings import settings

    key = settings.azure_speech_key.get_secret_value()
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto (dois niveis acima deste arquivo: src/config/settings.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Configuracao da aplicacao carregada via env vars e arquivo .env.

    Convencoes:
        - Chaves sensiveis usam `SecretStr` para evitar log acidental.
        - Caminhos sao convertidos para `Path` resolvido contra a raiz do projeto.
        - Toggles cloud-vs-local sao booleanos com default conservador (local).
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- Azure Speech ------------------------------------------------
    azure_speech_key: SecretStr = Field(default=SecretStr(""))
    azure_speech_region: str = Field(default="brazilsouth")

    # ----- Azure Language ----------------------------------------------
    azure_language_key: SecretStr = Field(default=SecretStr(""))
    azure_language_endpoint: str = Field(default="")

    # ----- Azure OpenAI ------------------------------------------------
    azure_openai_key: SecretStr = Field(default=SecretStr(""))
    azure_openai_endpoint: str = Field(default="")
    azure_openai_deployment: str = Field(default="gpt-4.1-mini")
    # Modelo multimodal (audio input) - deploy separado no Foundry,
    # ex: gpt-4o-mini-audio-preview. Quando vazio, pilar de emocao vocal
    # cai pra wav2vec2 local (com viés conhecido em PT-BR).
    azure_openai_audio_deployment: str = Field(default="")
    # Modelo multimodal (vision input) - deploy separado no Foundry,
    # ex: gpt-4o-mini ou gpt-4o. Quando vazio, pilar de emocao facial
    # cai pra FER local (com vies conhecido de FER-2013).
    azure_openai_vision_deployment: str = Field(default="")

    # ----- Toggles cloud vs local --------------------------------------
    use_cloud_transcription: bool = Field(default=False)

    # ----- Caminhos de modelos e indices -------------------------------
    yolo_weights_path: Path = Field(default=Path("models/yolov8n_surgical.pt"))
    rag_index_path: Path = Field(default=Path("data/processed/chroma"))

    # ----- Logging -----------------------------------------------------
    log_level: LogLevel = Field(default="INFO")

    @property
    def project_root(self) -> Path:
        """Raiz absoluta do projeto, util para resolver paths relativos."""
        return PROJECT_ROOT

    def yolo_weights_absolute(self) -> Path:
        """Retorna o caminho absoluto dos pesos YOLO."""
        return self._absolute(self.yolo_weights_path)

    def rag_index_absolute(self) -> Path:
        """Retorna o caminho absoluto do indice Chroma."""
        return self._absolute(self.rag_index_path)

    def _absolute(self, path: Path) -> Path:
        """Resolve `path` contra a raiz do projeto se for relativo."""
        return path if path.is_absolute() else PROJECT_ROOT / path


# Instancia singleton acessada por todo o projeto.
settings = Settings()
