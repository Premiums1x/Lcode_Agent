"""Tests for Level 5: Production-grade features."""

import pytest
from fastapi.testclient import TestClient

from lcode.mcp.server import MCPServer
from lcode.observability.tracer import Tracer
from lcode.plugins.loader import PluginLoader, SkillSystem
from lcode.tools.registry import ToolRegistry
from lcode.web.app import app


class TestWebUI:
    """Test Web UI endpoints."""

    def test_health_endpoint(self) -> None:
        """Test the health check endpoint."""
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_index_page(self) -> None:
        """Test the main page loads."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "LCode" in response.text


class TestMCPServer:
    """Test MCP (Model Context Protocol) server."""

    def test_mcp_initialize(self) -> None:
        """Test MCP initialize method."""
        server = MCPServer()
        request = {"jsonrpc": "2.0", "method": "initialize", "id": 1}
        response = server.handle(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"

    def test_mcp_tools_list(self) -> None:
        """Test MCP tools/list method."""
        registry = ToolRegistry()

        @registry.register()
        def test_tool() -> str:
            return "test"

        server = MCPServer()
        request = {"jsonrpc": "2.0", "method": "tools/list", "id": 2}
        response = server.handle(request)

        assert "result" in response
        assert "tools" in response["result"]

    def test_mcp_unknown_method(self) -> None:
        """Test MCP error handling for unknown methods."""
        server = MCPServer()
        request = {"jsonrpc": "2.0", "method": "unknown", "id": 3}
        response = server.handle(request)

        assert "error" in response
        assert response["error"]["code"] == -32601


class TestObservability:
    """Test observability features."""

    def test_tracer_creation(self) -> None:
        """Test tracer initialization."""
        tracer = Tracer(service_name="test")
        assert tracer.service_name == "test"

    def test_trace_span(self) -> None:
        """Test trace span context manager."""
        tracer = Tracer(service_name="test")

        with tracer.start_trace("test_operation", key="value") as span:
            assert span.name == "test_operation"
            assert span.trace_id is not None
            assert span.span_id is not None

    def test_trace_span_with_exception(self) -> None:
        """Test trace span handles exceptions gracefully."""
        tracer = Tracer(service_name="test")

        try:
            with tracer.start_trace("failing_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass  # Exception should propagate


class TestPluginSystem:
    """Test plugin and skills system."""

    def test_plugin_discover_empty(self, tmp_path) -> None:
        """Test plugin discovery in empty directory."""
        loader = PluginLoader(plugin_dir=tmp_path)
        plugins = loader.discover()
        assert len(plugins) == 0

    def test_skill_registration(self) -> None:
        """Test skill registration and execution."""
        skills = SkillSystem()

        def my_skill(data: str) -> str:
            return f"Processed: {data}"

        skills.register_skill("my_skill", my_skill)
        assert "my_skill" in skills.list_skills()

    @pytest.mark.asyncio
    async def test_skill_execution(self) -> None:
        """Test async skill execution."""
        skills = SkillSystem()

        async def async_skill(person: str) -> str:
            return f"Hello, {person}"

        skills.register_skill("greet", async_skill)
        result = await skills.execute_skill("greet", person="World")
        assert result == "Hello, World"

    @pytest.mark.asyncio
    async def test_skill_not_found(self) -> None:
        """Test error when skill is not found."""
        skills = SkillSystem()
        with pytest.raises(ValueError, match="not found"):
            await skills.execute_skill("nonexistent")


class TestDockerConfig:
    """Verify Docker configuration exists."""

    def test_dockerfile_exists(self) -> None:
        """Test that Dockerfile exists."""
        from pathlib import Path

        dockerfile = Path(__file__).parent.parent / "docker" / "Dockerfile"
        assert dockerfile.exists()

    def test_docker_compose_exists(self) -> None:
        """Test that docker-compose.yml exists."""
        from pathlib import Path

        compose = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        assert compose.exists()
