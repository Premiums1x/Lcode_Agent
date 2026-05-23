"""Tests for Level 4: Multi-Agent Framework."""

import pytest

from lcode.agents.chat_agent import ChatAgent
from lcode.llm.base import LLMResponse, Message
from lcode.orchestration.manager import AgentManager, Task


class MockLLMProvider:
    """Mock LLM for testing multi-agent orchestration."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses = responses or []
        self.call_index = 0

    async def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        return LLMResponse(content="Mock answer.", model="mock")

    async def chat_stream(self, messages: list[Message], **kwargs):
        yield "Mock"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]


class TestTask:
    """Test Task data model."""

    def test_task_creation(self) -> None:
        """Test creating a task."""
        task = Task("1", "Do something", "agent1")
        assert task.task_id == "1"
        assert task.description == "Do something"
        assert task.assignee == "agent1"
        assert task.status == "pending"
        assert task.result is None

    def test_task_to_dict(self) -> None:
        """Test task serialization."""
        task = Task("1", "Do something")
        task.status = "completed"
        task.result = "Done"

        d = task.to_dict()
        assert d["task_id"] == "1"
        assert d["status"] == "completed"
        assert d["result"] == "Done"


class TestAgentManager:
    """Test the Agent Manager / orchestrator."""

    def test_register_agent(self) -> None:
        """Test agent registration."""
        llm = MockLLMProvider()
        manager = AgentManager(llm=llm)
        agent = ChatAgent(name="test", llm=llm)

        manager.register_agent(agent)
        assert "test" in manager.agents

    def test_unregister_agent(self) -> None:
        """Test agent unregistration."""
        llm = MockLLMProvider()
        manager = AgentManager(llm=llm)
        agent = ChatAgent(name="test", llm=llm)

        manager.register_agent(agent)
        manager.unregister_agent("test")
        assert "test" not in manager.agents

    @pytest.mark.asyncio
    async def test_execute_single_task(self) -> None:
        """Test executing a single task."""
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content='{"tasks": [{"id": "1", "description": "Say hello", "agent": "any"}]}',
                    model="mock",
                ),
            ]
        )
        manager = AgentManager(llm=llm)
        agent = ChatAgent(name="greeter", llm=llm)
        manager.register_agent(agent)

        response = await manager.execute("Say hello")
        assert isinstance(response, LLMResponse)
        assert len(manager.tasks) == 1
        assert manager.tasks[0].status in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_plan_decomposition(self) -> None:
        """Test task decomposition planning."""
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content='{"tasks": [{"id": "1", "description": "Step 1"}, {"id": "2", "description": "Step 2"}]}',
                    model="mock",
                ),
            ]
        )
        manager = AgentManager(llm=llm)
        agent = ChatAgent(name="worker", llm=llm)
        manager.register_agent(agent)

        await manager.execute("Complex task")
        assert len(manager.tasks) == 2


class TestAgentCommunication:
    """Test agent communication via event bus."""

    @pytest.mark.asyncio
    async def test_event_bus_publish(self) -> None:
        """Test basic event bus publish/subscribe."""
        from lcode.core.events import EventBus

        bus = EventBus()
        received = []

        def handler(**kwargs) -> None:
            received.append(kwargs)

        bus.subscribe("test_event", handler)
        await bus.publish("test_event", message="hello")

        assert len(received) == 1
        assert received[0]["message"] == "hello"

    @pytest.mark.asyncio
    async def test_event_bus_multiple_subscribers(self) -> None:
        """Test multiple subscribers receive events."""
        from lcode.core.events import EventBus

        bus = EventBus()
        count = [0]

        def handler1(**kwargs) -> None:
            count[0] += 1

        def handler2(**kwargs) -> None:
            count[0] += 1

        bus.subscribe("event", handler1)
        bus.subscribe("event", handler2)
        await bus.publish("event")

        assert count[0] == 2
