"""IM (Instant Messaging) protocol adapters.

Provides adapters for common IM platforms:
- Webhook (generic HTTP endpoint)
- WebSocket (real-time bidirectional)
"""

import json
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from lcode.agents.chat_agent import ChatAgent
from lcode.llm.openai_provider import OpenAIProvider

router = APIRouter(prefix="/im", tags=["IM"])


class IMMessage:
    """Standardized IM message format."""

    def __init__(
        self,
        sender: str,
        content: str,
        platform: str = "generic",
        channel: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.sender = sender
        self.content = content
        self.platform = platform
        self.channel = channel
        self.timestamp = timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "content": self.content,
            "platform": self.platform,
            "channel": self.channel,
            "timestamp": self.timestamp,
        }


class IMAdapter:
    """Base class for IM adapters."""

    def __init__(self, agent: ChatAgent | None = None) -> None:
        self.agent = agent

    async def handle_message(self, message: IMMessage) -> str:
        """Handle an incoming IM message and return a response."""
        if not self.agent:
            return "Agent not configured."

        from lcode.observability.tracer import tracer

        with tracer.start_trace("im_message", platform=message.platform, sender=message.sender):
            response = await self.agent.run(message.content)
            return response.content

    async def send_message(self, recipient: str, content: str, **kwargs: Any) -> None:
        """Send a message to a recipient. Override in subclass."""
        raise NotImplementedError


class WebhookAdapter(IMAdapter):
    """Generic webhook adapter for IM platforms.

    Accepts POST requests with JSON payload:
    {
        "sender": "user123",
        "content": "Hello bot",
        "platform": "custom",
        "channel": "general"
    }
    """

    async def receive(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Receive and process a webhook payload."""
        message = IMMessage(
            sender=payload.get("sender", "unknown"),
            content=payload.get("content", ""),
            platform=payload.get("platform", "webhook"),
            channel=payload.get("channel"),
        )

        response_text = await self.handle_message(message)

        return {
            "status": "ok",
            "response": response_text,
            "platform": message.platform,
        }


# Global webhook adapter instance
webhook_adapter = WebhookAdapter()


@router.post("/webhook")
async def im_webhook(request: Request) -> dict[str, Any]:
    """Generic webhook endpoint for IM integration."""
    payload = await request.json()
    return await webhook_adapter.receive(payload)


@router.websocket("/ws")
async def im_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time IM communication."""
    await websocket.accept()

    # Create a per-connection agent
    llm = OpenAIProvider()
    agent = ChatAgent(name="im_bot", llm=llm)
    adapter = WebhookAdapter(agent=agent)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                message = IMMessage(
                    sender=payload.get("sender", "websocket_user"),
                    content=payload.get("content", data),
                    platform=payload.get("platform", "websocket"),
                )
                response = await adapter.handle_message(message)
                await websocket.send_json({
                    "type": "response",
                    "content": response,
                })
            except json.JSONDecodeError:
                # Treat raw text as message content
                message = IMMessage(sender="websocket_user", content=data, platform="websocket")
                response = await adapter.handle_message(message)
                await websocket.send_json({
                    "type": "response",
                    "content": response,
                })
    except WebSocketDisconnect:
        pass
