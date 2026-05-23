"""Base Agent class."""

from abc import ABC, abstractmethod
from typing import Any

from lcode.core.events import event_bus
from lcode.llm.base import BaseLLMProvider, LLMResponse, Message


class BaseAgent(ABC):
    """Base class for all agents in LCode."""

    def __init__(
        self,
        name: str,
        llm: BaseLLMProvider,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.message_history: list[Message] = []
        self._setup_events()

    def _setup_events(self) -> None:
        """Subscribe to relevant events. Override in subclass."""
        pass

    @abstractmethod
    async def run(self, user_input: str, **kwargs: Any) -> LLMResponse:
        """Process user input and return a response.

        Args:
            user_input: The user's message.
            **kwargs: Additional context.

        Returns:
            LLM response.
        """
        ...

    def reset(self) -> None:
        """Clear conversation history."""
        self.message_history.clear()

    async def _call_llm(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Internal helper to call the LLM."""
        return await self.llm.chat(messages=messages, tools=tools, **kwargs)

    def _build_messages(self, user_input: str) -> list[Message]:
        """Build the message list including system prompt and history."""
        messages = [Message(role="system", content=self.system_prompt)]
        messages.extend(self.message_history)
        messages.append(Message(role="user", content=user_input))
        return messages

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to history."""
        self.message_history.append(Message(role=role, content=content))
