"""Configuration management for VideoTagger.

Loads and validates credentials from environment variables and .env files.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SynologyConfig(BaseSettings):
    """Synology NAS connection credentials."""

    model_config = SettingsConfigDict(env_prefix="SYNOLOGY_")

    host: str = Field(..., description="Synology NAS hostname or IP")
    user: str = Field(..., description="Synology username")
    password: str = Field(..., description="Synology password")
    video_path: str = Field(..., description="Path to video folder on NAS")


class AirtableConfig(BaseSettings):
    """Airtable API credentials."""

    model_config = SettingsConfigDict(env_prefix="AIRTABLE_")

    api_key: str = Field(..., description="Airtable API key (pat...)")
    base_id: str = Field(..., description="Airtable base ID (app...)")
    table_id: str = Field(..., description="Airtable table ID (tbl...)")


class RunPodS3Config(BaseSettings):
    """RunPod S3-compatible storage credentials."""

    model_config = SettingsConfigDict(env_prefix="RUNPOD_S3_")

    endpoint: str = Field(..., description="S3 API endpoint URL")
    bucket: str = Field(..., description="S3 bucket name")
    access_key: str = Field(..., description="S3 access key")
    secret_key: str = Field(..., description="S3 secret key")


class RunPodSSHConfig(BaseSettings):
    """RunPod SSH connection credentials (deprecated - use API instead)."""

    model_config = SettingsConfigDict(env_prefix="RUNPOD_SSH_")

    host: str = Field(default="", description="SSH host (deprecated)")
    user: str = Field(default="", description="SSH username (deprecated)")
    key_path: Path | None = Field(default=None, description="Path to SSH private key (deprecated)")
    pod_id: str = Field(
        default="",
        description="Preferred pod ID (optional - will auto-detect if not set)",
    )

    @field_validator("key_path", mode="before")
    @classmethod
    def expand_and_validate_path(cls, v: str | Path | None) -> Path | None:
        """Expand ~ in path and validate the file exists."""
        if v is None or v == "":
            return None
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(f"SSH key file not found: {path}")
        if not path.is_file():
            raise ValueError(f"SSH key path is not a file: {path}")
        return path


class RunPodAPIConfig(BaseSettings):
    """RunPod API configuration."""

    model_config = SettingsConfigDict(env_prefix="RUNPOD_")

    api_key: str = Field(..., description="RunPod API key")


class LLMConfig(BaseSettings):
    """LLM (vLLM) configuration for video analysis."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    endpoint: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM OpenAI-compatible API endpoint",
    )
    model: str = Field(
        default="Qwen/Qwen3-VL-4B-Instruct",
        description="Model name for vision-language analysis",
    )
    api_key: str = Field(
        default="EMPTY",
        description="API key (use 'EMPTY' for local vLLM)",
    )
    frame_count: int = Field(
        default=8,
        description="Number of frames to extract from video",
        ge=1,
        le=32,
    )
    frame_max_size: int = Field(
        default=768,
        description="Maximum frame dimension in pixels (reduces context usage)",
        ge=256,
        le=1920,
    )


class AudioConfig(BaseSettings):
    """Audio analysis pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="AUDIO_")

    enabled: bool = Field(
        default=True,
        description="Enable audio analysis pipeline",
    )
    vad_threshold: float = Field(
        default=0.5,
        description="Voice activity detection probability threshold",
        ge=0.0,
        le=1.0,
    )
    min_speech_duration_ms: int = Field(
        default=250,
        description="Minimum speech segment duration in milliseconds",
        ge=50,
    )
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate for processing (16kHz optimal for speech)",
    )


class Settings(BaseSettings):
    """Root settings aggregating all configuration groups."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    synology: SynologyConfig = Field(default_factory=SynologyConfig)
    airtable: AirtableConfig = Field(default_factory=AirtableConfig)
    runpod_s3: RunPodS3Config = Field(default_factory=RunPodS3Config)
    runpod_ssh: RunPodSSHConfig = Field(default_factory=RunPodSSHConfig)
    runpod_api: RunPodAPIConfig = Field(default_factory=RunPodAPIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Validated settings loaded from environment.

    Raises:
        ValidationError: If required credentials are missing or invalid.
    """
    return Settings()


def mask_credential(value: str) -> str:
    """Mask a credential for display, showing first and last 4 characters.

    Args:
        value: The credential string to mask.

    Returns:
        Masked string like "patN...0e4e" or full string if too short.
    """
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"
