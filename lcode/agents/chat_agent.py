"""Simple chat agent with conversation memory."""

from typing import Any

from lcode.agents.base import BaseAgent
from lcode.llm.base import LLMResponse


class ChatAgent(BaseAgent):
    """A simple conversational agent supporting continuous chat.

    This is the Level 1 implementation - a single agent that
    maintains conversation history.
    """

    def _setup_events(self) -> None:
        """Subscribe to relevant events."""
        pass

    async def run(self, user_input: str, **kwargs: Any) -> LLMResponse:
        """Process user input and return a response.

        Args:
            user_input: The user's message.
            **kwargs: Extra arguments passed to LLM.

        Returns:
            LLM response.
        """
        # Build messages with history
        messages = self._build_messages(user_input)

        # Call LLM
        response = await self._call_llm(messages, **kwargs)

        # Update history
        self.add_to_history("user", user_input)
        self.add_to_history("assistant", response.content)

        return response
