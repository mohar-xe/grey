from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Resolve .env path relative to this file (grey/.env)
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class NvidiaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    api_key: str = Field(..., alias="NVIDIA_API_KEY")
    base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    model: str = Field(
        default="meta/llama-3.1-70b-instruct",
        alias="LLM_MODEL",
    )
    temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        alias="EMBED_MODEL",
    )
    dimension: int = Field(default=1024, alias="EMBED_DIMENSION")
    batch_size: int = Field(default=50, alias="EMBED_BATCH_SIZE")


class VLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    model: str = Field(
        default="meta/llama-3.2-90b-vision-instruct",
        alias="VLM_MODEL",
    )
    max_tokens: int = Field(default=1024, alias="VLM_MAX_TOKENS")


class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    model: str = Field(
        default="nvidia/parakeet-ctc-1.1b-asr",
        alias="STT_MODEL",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    nvidia: NvidiaSettings = NvidiaSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vlm: VLMSettings = VLMSettings()
    stt: STTSettings = STTSettings()

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()