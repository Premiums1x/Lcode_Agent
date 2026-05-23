"""Tool registry for managing and calling tools."""

import inspect
import json
from collections.abc import Callable
from typing import Any


class Tool:
    """Represents a registered tool."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def call(self, **kwargs: Any) -> Any:
        """Execute the tool function."""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Decorator to register a function as a tool.

        Args:
            name: Tool name. Uses function name if not provided.
            description: Tool description. Uses docstring if not provided.
            parameters: JSON Schema for parameters. Auto-generated if not provided.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "No description provided.")
            tool_params = parameters or self._infer_parameters(func)

            self._tools[tool_name] = Tool(
                name=tool_name,
                description=tool_desc,
                func=func,
                parameters=tool_params,
            )
            return func

        return decorator

    def _infer_parameters(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Infer JSON Schema from function signature."""
        sig = inspect.signature(func)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = param.annotation
            json_type = "string"
            if param_type in (int,):
                json_type = "integer"
            elif param_type in (float,):
                json_type = "number"
            elif param_type in (bool,):
                json_type = "boolean"
            elif param_type in (list,):
                json_type = "array"
            elif param_type in (dict,):
                json_type = "object"

            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """Return all tools in OpenAI format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: str | dict[str, Any]) -> Any:
        """Execute a tool by name with arguments.

        Args:
            name: Tool name.
            arguments: JSON string or dict of arguments.

        Returns:
            Tool execution result.
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found.")

        kwargs = json.loads(arguments) if isinstance(arguments, str) else arguments

        return await tool.call(**kwargs)


# Global tool registry instance
tool_registry = ToolRegistry()
