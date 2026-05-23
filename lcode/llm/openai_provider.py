"""OpenAI-compatible LLM provider.

Supports OpenAI, DeepSeek, and any other OpenAI-compatible API.
"""

from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI

from lcode.core.config import settings
from lcode.llm.base import BaseLLMProvider, LLMResponse, Message


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> None:
        config = settings.effective_llm_config
        self.api_key = api_key or config["api_key"]
        self.base_url = base_url or config["base_url"]
        self.default_model = default_model or config["model"]

        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY or DEEPSEEK_API_KEY in environment."
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.AsyncClient(timeout=120.0),
        )

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request."""
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(kwargs)

        response = await self.client.chat.completions.create(**payload)

        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            tool_calls=tool_calls,
            raw=response,
        )

    async def chat_stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        payload.update(kwargs)

        stream = await self.client.chat.completions.create(**payload)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings using OpenAI embedding API.

        Falls back to sentence-transformers if no embedding model is specified.
        """
        # Use a small local model via sentence-transformers for embeddings
        # to avoid dependency on OpenAI embedding API costs
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(settings.embedding_model)
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        result: list[list[float]] = embeddings.tolist()
        return result
