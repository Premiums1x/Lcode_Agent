"""MCP (Model Context Protocol) server implementation.

MCP is a protocol for connecting AI systems with external data sources and tools.
This implementation provides a simplified MCP-compatible server.

Reference: https://modelcontextprotocol.io/
"""

import json
from typing import Any

from fastapi import APIRouter, Request, Response

from lcode.tools.registry import tool_registry

router = APIRouter(prefix="/mcp", tags=["MCP"])


class MCPServer:
    """Simplified MCP server implementation.

    MCP follows a JSON-RPC-like request/response model:
    - initialize: Handshake
    - tools/list: List available tools
    - tools/call: Execute a tool
    """

    def __init__(self) -> None:
        self.protocol_version = "2024-11-05"
        self.server_info = {"name": "lcode-mcp", "version": "0.1.0"}

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle an MCP request.

        Args:
            request: JSON-RPC-like request dict.

        Returns:
            Response dict.
        """
        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            return self._initialize(req_id)
        elif method == "tools/list":
            return self._list_tools(req_id)
        elif method == "tools/call":
            return self._call_tool(req_id, request.get("params", {}))
        else:
            return self._error(req_id, -32601, f"Method not found: {method}")

    def _initialize(self, req_id: Any) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": self.protocol_version,
                "serverInfo": self.server_info,
                "capabilities": {"tools": {}},
            },
        }

    def _list_tools(self, req_id: Any) -> dict[str, Any]:
        tools = []
        for tool in tool_registry.list_tools():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            })
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools},
        }

    def _call_tool(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        import asyncio
        try:
            result = asyncio.run(tool_registry.execute(tool_name, arguments))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(result)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True,
                },
            }

    def _error(self, req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


# Global MCP server instance
mcp_server = MCPServer()


@router.post("/")
async def mcp_endpoint(request: Request) -> Response:
    """MCP JSON-RPC endpoint."""
    body = await request.json()
    response = mcp_server.handle(body)
    return Response(
        content=json.dumps(response),
        media_type="application/json",
    )
