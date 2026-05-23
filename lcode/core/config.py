"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """LCode application settings.

    Priority: env vars > .env file > defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI-compatible API base URL")
    default_model: str = Field(default="gpt-4o-mini", description="Default LLM model")
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=4096, ge=1)

    # DeepSeek Support (uses OpenAI-compatible API)
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", description="DeepSeek API base URL")

    # Framework Settings
    app_name: str = Field(default="LCode")
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # Memory & RAG
    memory_type: Literal["in_memory", "sqlite", "redis"] = Field(default="sqlite")
    memory_db_path: Path = Field(default=Path("./data/lcode_memory.db"))
    vector_db_path: Path = Field(default=Path("./data/vector_db"))
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", description="Sentence-transformers model for embeddings")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    top_k: int = Field(default=5)

    # Web UI
    web_host: str = Field(default="0.0.0.0")
    web_port: int = Field(default=8000)
    web_reload: bool = Field(default=False)

    # Multi-Agent
    max_agents: int = Field(default=10, ge=1)
    agent_timeout: int = Field(default=120, ge=1, description="Agent task timeout in seconds")

    # MCP
    mcp_enabled: bool = Field(default=True)
    mcp_server_port: int = Field(default=8080)

    # Plugin
    plugin_dir: Path = Field(default=Path("./plugins"))
    auto_discover_plugins: bool = Field(default=True)

    @property
    def effective_llm_config(self) -> dict:
        """Return the active LLM configuration.

        Priority: DeepSeek > OpenAI
        """
        if self.deepseek_api_key:
            return {
                "api_key": self.deepseek_api_key,
                "base_url": self.deepseek_base_url,
                "model": self.default_model,
            }
        return {
            "api_key": self.openai_api_key,
            "base_url": self.openai_base_url,
            "model": self.default_model,
        }

    @property
    def data_dir(self) -> Path:
        """Return the data directory, creating it if necessary."""
        path = Path("./data")
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
