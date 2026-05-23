"""Base LLM provider abstraction."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None  # for tool messages
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to provider-native format."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    raw: Any | None = None  # provider-specific raw response


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    default_model: str = ""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of conversation messages.
            model: Model identifier. Uses default if None.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Optional tool definitions for function calling.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Standardized LLM response.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        Yields:
            Token strings as they are generated.
        """
        ...

    @abstractmethod
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of strings to embed.
            model: Embedding model identifier.

        Returns:
            List of embedding vectors.
        """
        ...
