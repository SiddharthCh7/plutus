"""Configuration management for Plutus.

Loads and validates configuration from YAML files and environment variables.
Designed to be portable for future cloud migration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_project_root() -> Path:
    """Get the project root directory.
    
    Path: __file__ is config.py
    -> parent is plutus/
    -> parent is src/
    -> parent is plutus (project root)
    """
    return Path(__file__).parent.parent.parent


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = get_project_root() / "config" / filename
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


class QdrantSettings(BaseSettings):
    """Qdrant vector database settings (portable: local or cloud)."""
    
    model_config = SettingsConfigDict(env_prefix="QDRANT_")
    
    host: str = "localhost"
    port: int = 6333
    api_key: str | None = None  # For cloud deployment
    collection_name: str = "plutus_memory"
    
    @property
    def is_cloud(self) -> bool:
        """Check if using cloud Qdrant."""
        return self.api_key is not None


class LangfuseSettings(BaseSettings):
    """Langfuse observability settings."""
    
    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")
    
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    enabled: bool = True
    
    @property
    def is_configured(self) -> bool:
        """Check if Langfuse is properly configured."""
        return bool(self.public_key and self.secret_key)


class EmailSettings(BaseSettings):
    """Email notification settings (modular, cloud-portable)."""
    
    model_config = SettingsConfigDict(env_prefix="SMTP_")
    
    host: str = "smtp.gmail.com"
    port: int = 587
    user: str = ""
    password: str = ""
    
    @property
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(self.user and self.password)


class LoggingSettings(BaseSettings):
    """Structured logging settings (cloud-portable)."""
    
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = "INFO"
    format: str = "json"  # 'json' for cloud, 'console' for local
    file_path: str | None = None  # Optional file logging
    
    @property
    def is_json(self) -> bool:
        """Check if using JSON format (cloud-ready)."""
        return self.format.lower() == "json"


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Keys
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    
    # Notification target
    notification_email: str = Field(default="", alias="NOTIFICATION_EMAIL")
    
    # Sub-settings
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Paths
    @property
    def project_root(self) -> Path:
        return get_project_root()
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def portfolio_path(self) -> Path:
        return self.data_dir / "portfolio.json"
    
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


class ModelConfig:
    """Model router configuration from models.yaml."""
    
    def __init__(self) -> None:
        self._config = load_yaml_config("models.yaml")
    
    @property
    def default_provider(self) -> str:
        return self._config.get("default_provider", "gemini")
    
    @property
    def default_model(self) -> str:
        return self._config.get("default_model", "gemini-2.5-flash")
    
    @property
    def providers(self) -> dict[str, Any]:
        return self._config.get("providers", {})
    
    def get_agent_model(self, agent_name: str) -> str | None:
        """Get model override for specific agent."""
        agent_models = self._config.get("agent_models", {})
        return agent_models.get(agent_name)
    
    @property
    def summarization_config(self) -> dict[str, str]:
        """Get summarization model config."""
        return self._config.get("summarization", {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
        })


class ScheduleConfig:
    """Task scheduler configuration from schedule.yaml."""
    
    def __init__(self) -> None:
        self._config = load_yaml_config("schedule.yaml")
    
    @property
    def tasks(self) -> dict[str, Any]:
        return self._config.get("tasks", {})
    
    @property
    def timezone(self) -> str:
        return self._config.get("timezone", "Asia/Kolkata")
    
    @property
    def settings(self) -> dict[str, Any]:
        return self._config.get("settings", {})
    
    def get_task(self, task_name: str) -> dict[str, Any] | None:
        """Get configuration for a specific task."""
        return self.tasks.get(task_name)


# Singleton instances
_settings: Settings | None = None
_model_config: ModelConfig | None = None
_schedule_config: ScheduleConfig | None = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_model_config() -> ModelConfig:
    """Get model router configuration (singleton)."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config


def get_schedule_config() -> ScheduleConfig:
    """Get scheduler configuration (singleton)."""
    global _schedule_config
    if _schedule_config is None:
        _schedule_config = ScheduleConfig()
    return _schedule_config
