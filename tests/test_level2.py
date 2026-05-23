"""Tests for Level 2: Tool Calling Agent."""

import pytest

from lcode.agents.react_agent import ReActAgent
from lcode.llm.base import LLMResponse, Message
from lcode.tools.registry import ToolRegistry


class MockLLMProvider:
    """Mock LLM that can simulate tool calls."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses = responses or []
        self.call_index = 0
        self.last_tools = None

    async def chat(self, messages: list[Message], tools=None, **kwargs) -> LLMResponse:
        self.last_tools = tools
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
            return response
        return LLMResponse(content="Final answer.", model="mock")

    async def chat_stream(self, messages: list[Message], **kwargs):
        yield "Final"
        yield " answer."

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]


class TestToolRegistry:
    """Test the tool registry system."""

    def test_register_tool(self) -> None:
        """Test tool registration."""
        registry = ToolRegistry()

        @registry.register()
        def test_tool(x: int) -> int:
            """A test tool."""
            return x * 2

        tool = registry.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        assert "test tool" in tool.description.lower()

    def test_list_tools(self) -> None:
        """Test listing registered tools."""
        registry = ToolRegistry()

        @registry.register()
        def tool_a() -> str:
            return "a"

        @registry.register()
        def tool_b() -> str:
            return "b"

        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}

    def test_openai_schema(self) -> None:
        """Test conversion to OpenAI function schema."""
        registry = ToolRegistry()

        @registry.register()
        def calculate(a: int, b: int) -> int:
            """Calculate something."""
            return a + b

        schemas = registry.to_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "calculate"

    @pytest.mark.asyncio
    async def test_execute_tool(self) -> None:
        """Test tool execution."""
        registry = ToolRegistry()

        @registry.register()
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await registry.execute("greet", {"name": "World"})
        assert result == "Hello, World!"


class TestReActAgent:
    """Test the ReAct agent implementation."""

    @pytest.mark.asyncio
    async def test_react_without_tools(self) -> None:
        """Test ReAct agent without tool calls."""
        llm = MockLLMProvider(
            [
                LLMResponse(content="Simple answer.", model="mock"),
            ]
        )
        registry = ToolRegistry()
        agent = ReActAgent(name="test", llm=llm, tool_registry=registry)

        response = await agent.run("Hello")
        assert response.content == "Simple answer."

    @pytest.mark.asyncio
    async def test_react_with_tool_call(self) -> None:
        """Test ReAct agent with a tool call."""
        registry = ToolRegistry()

        @registry.register()
        def get_number() -> str:
            return "42"

        # First response: tool call, Second: final answer
        llm = MockLLMProvider(
            [
                LLMResponse(
                    content="I'll check.",
                    model="mock",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_number",
                                "arguments": "{}",
                            },
                        }
                    ],
                ),
                LLMResponse(content="The number is 42.", model="mock"),
            ]
        )

        agent = ReActAgent(name="test", llm=llm, tool_registry=registry)
        response = await agent.run("What is the number?")

        assert "42" in response.content
        assert llm.call_index == 2


class TestBuiltinTools:
    """Test built-in tools."""

    def test_calculator(self) -> None:
        """Test the calculator tool."""
        from lcode.tools.builtin import calculator

        result = calculator("2 + 2")
        assert "4" in result

        result = calculator("sqrt(16)")
        assert "4.0" in result

        result = calculator("1 / 0")
        assert "Error" in result
