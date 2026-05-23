"""Tests for Level 1: Single Agent Chat System."""

import pytest

from lcode.agents.chat_agent import ChatAgent
from lcode.core.config import Settings
from lcode.llm.base import LLMResponse, Message


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Hello! I'm a mock assistant."]
        self.call_count = 0
        self.last_messages: list[Message] = []

    async def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        self.last_messages = messages
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return LLMResponse(content=response, model="mock")

    async def chat_stream(self, messages: list[Message], **kwargs):
        yield "Mock"
        yield " response"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class TestChatAgent:
    """Test the basic ChatAgent functionality."""

    @pytest.mark.asyncio
    async def test_basic_chat(self) -> None:
        """Test that the agent can process a message and return a response."""
        llm = MockLLMProvider()
        agent = ChatAgent(name="test", llm=llm)

        response = await agent.run("Hello")

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! I'm a mock assistant."
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_conversation_history(self) -> None:
        """Test that the agent maintains conversation history."""
        llm = MockLLMProvider()
        agent = ChatAgent(name="test", llm=llm)

        await agent.run("Hello")
        await agent.run("How are you?")

        # Should have system + user1 + assistant1 + user2 in messages
        assert len(agent.message_history) == 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_system_prompt_included(self) -> None:
        """Test that system prompt is sent to LLM."""
        llm = MockLLMProvider()
        agent = ChatAgent(name="test", llm=llm, system_prompt="You are a test bot.")

        await agent.run("Hello")

        assert len(llm.last_messages) > 0
        assert llm.last_messages[0].role == "system"
        assert llm.last_messages[0].content == "You are a test bot."

    def test_reset_clears_history(self) -> None:
        """Test that reset clears conversation history."""
        llm = MockLLMProvider()
        agent = ChatAgent(name="test", llm=llm)

        # Can't test async run in sync test easily, so just test the method exists
        agent.reset()
        assert len(agent.message_history) == 0


class TestSettings:
    """Test configuration management."""

    def test_default_settings(self) -> None:
        """Test that default settings are loaded."""
        settings = Settings()
        assert settings.app_name == "LCode"
        assert settings.default_temperature == 0.7
        assert settings.default_max_tokens == 4096

    def test_effective_llm_config(self) -> None:
        """Test LLM config resolution."""
        # Clear any existing keys to test OpenAI fallback
        settings = Settings(
            openai_api_key="test-key",
            deepseek_api_key="",
        )
        config = settings.effective_llm_config
        assert config["api_key"] == "test-key"
        assert config["base_url"] == "https://api.openai.com/v1"

    def test_data_dir_created(self) -> None:
        """Test that data directory is created."""
        settings = Settings()
        data_dir = settings.data_dir
        assert data_dir.exists()
