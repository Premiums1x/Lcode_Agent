"""FastAPI web application for LCode.

Provides:
- Chat interface
- Agent management
- Document ingestion
- Real-time streaming via WebSocket
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from lcode.agents.chat_agent import ChatAgent
from lcode.agents.rag_agent import RAGAgent
from lcode.agents.react_agent import ReActAgent
from lcode.core.config import settings
from lcode.llm.openai_provider import OpenAIProvider
from lcode.mcp.im_adapter import router as im_router
from lcode.mcp.server import router as mcp_router
from lcode.observability.tracer import tracer
from lcode.tools.registry import tool_registry


# In-memory agent instances (in production, use proper state management)
_agent_instances: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"🚀 {settings.app_name} Web UI starting on http://{settings.web_host}:{settings.web_port}")
    yield
    # Shutdown
    print("👋 Shutting down")


app = FastAPI(
    title=settings.app_name,
    description="LCode AI Agent Framework Web Interface",
    version="0.1.0",
    lifespan=lifespan,
)

# Include MCP and IM routers
app.include_router(mcp_router)
app.include_router(im_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to mount static files
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the main UI page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LCode - AI Agent Framework</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f5f5f7; color: #333; height: 100vh; display: flex;
            }
            .sidebar {
                width: 240px; background: #1a1a2e; color: #fff; padding: 20px;
                display: flex; flex-direction: column; gap: 10px; flex-shrink: 0;
            }
            .sidebar h1 { font-size: 1.5rem; margin-bottom: 20px; }
            .sidebar button {
                background: #16213e; color: #fff; border: 1px solid #0f3460;
                padding: 12px; border-radius: 8px; cursor: pointer; text-align: left;
                transition: background 0.2s;
            }
            .sidebar button:hover { background: #0f3460; }
            .sidebar button.active { background: #e94560; border-color: #e94560; }
            .main { flex: 1; display: flex; flex-direction: column; padding: 20px; min-width: 0; }
            .chat-container {
                flex: 1; background: #fff; border-radius: 12px; padding: 20px;
                overflow-y: auto; display: flex; flex-direction: column; gap: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .message {
                max-width: 80%; padding: 12px 16px; border-radius: 12px;
                line-height: 1.5; word-wrap: break-word;
            }
            .message.user { align-self: flex-end; background: #e94560; color: #fff; }
            .message.assistant { align-self: flex-start; background: #f0f0f5; color: #333; }
            .input-area {
                margin-top: 15px; display: flex; gap: 10px;
            }
            .input-area input {
                flex: 1; padding: 14px 18px; border: 1px solid #ddd;
                border-radius: 10px; font-size: 1rem; outline: none;
            }
            .input-area input:focus { border-color: #e94560; }
            .input-area button {
                padding: 14px 24px; background: #e94560; color: #fff;
                border: none; border-radius: 10px; cursor: pointer; font-size: 1rem;
            }
            .input-area button:hover { background: #d63850; }
            .status { font-size: 0.85rem; color: #888; margin-top: 5px; }
            pre { background: #f8f8f8; padding: 10px; border-radius: 6px; overflow-x: auto; }
            code { font-family: 'Consolas', monospace; font-size: 0.9rem; }
            .info-panel {
                width: 260px; background: #fff; border-left: 1px solid #e0e0e0;
                padding: 20px; display: flex; flex-direction: column; gap: 20px;
                flex-shrink: 0; overflow-y: auto;
            }
            .info-section h3 {
                font-size: 0.9rem; color: #888; text-transform: uppercase;
                letter-spacing: 0.05em; margin-bottom: 10px; border-bottom: 1px solid #eee;
                padding-bottom: 6px;
            }
            .info-item {
                display: flex; justify-content: space-between; padding: 6px 0;
                font-size: 0.9rem; border-bottom: 1px solid #f5f5f5;
            }
            .info-item:last-child { border-bottom: none; }
            .info-label { color: #888; }
            .info-value { font-weight: 600; color: #333; }
            .info-value.model { color: #e94560; }
            .info-value.status { color: #28a745; }
            .info-value.status.disconnected { color: #dc3545; }
            .tool-item {
                padding: 8px 0; border-bottom: 1px solid #f5f5f5;
            }
            .tool-item:last-child { border-bottom: none; }
            .tool-name { font-weight: 600; font-size: 0.85rem; color: #1a1a2e; }
            .tool-desc { font-size: 0.8rem; color: #888; margin-top: 2px; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h1>LCode</h1>
            <button id="btn-chat" class="active" onclick="setMode('chat')">Chat Agent</button>
            <button id="btn-react" onclick="setMode('react')">ReAct Agent (Tools)</button>
            <button id="btn-rag" onclick="setMode('rag')">RAG Agent (Docs)</button>
            <div style="margin-top: auto; font-size: 0.8rem; color: #888;">
                v0.1.0 | Level 5 Ready
            </div>
        </div>
        <div class="main">
            <div class="chat-container" id="chat"></div>
            <div class="status" id="status">Ready</div>
            <div class="input-area">
                <input type="text" id="msgInput" placeholder="Type your message..." onkeydown="if(event.key==='Enter')send()">
                <button onclick="send()">Send</button>
            </div>
        </div>
        <div class="info-panel" id="infoPanel">
            <div class="info-section">
                <h3>Session Info</h3>
                <div class="info-item">
                    <span class="info-label">Agent</span>
                    <span class="info-value" id="infoAgent">—</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Model</span>
                    <span class="info-value model" id="infoModel">—</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Status</span>
                    <span class="info-value status" id="infoStatus">Disconnected</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Messages</span>
                    <span class="info-value" id="infoMessages">0</span>
                </div>
            </div>
            <div class="info-section">
                <h3>Tools</h3>
                <div id="infoTools"><div class="tool-item"><div class="tool-desc">No tools</div></div></div>
            </div>
        </div>
        <script>
            let ws = null;
            let mode = 'chat';
            let messageCount = 0;
            const chat = document.getElementById('chat');
            const status = document.getElementById('status');
            const infoAgent = document.getElementById('infoAgent');
            const infoModel = document.getElementById('infoModel');
            const infoStatus = document.getElementById('infoStatus');
            const infoMessages = document.getElementById('infoMessages');
            const infoTools = document.getElementById('infoTools');

            function connect() {
                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${proto}//${window.location.host}/ws/${mode}`);
                ws.onopen = () => {
                    status.innerText = 'Connected';
                    infoStatus.innerText = 'Connected';
                    infoStatus.classList.remove('disconnected');
                    infoAgent.innerText = mode.toUpperCase();
                };
                ws.onmessage = (e) => {
                    const data = JSON.parse(e.data);
                    if (data.type === 'config') {
                        infoModel.innerText = data.model || '—';
                        renderTools(data.tools || []);
                    } else if (data.type === 'stream') {
                        appendMsg(data.content, 'assistant', true);
                    } else if (data.type === 'message') {
                        appendMsg(data.content, 'assistant', false);
                        infoModel.innerText = data.model || infoModel.innerText;
                        messageCount++;
                        infoMessages.innerText = messageCount;
                    } else if (data.type === 'error') {
                        appendMsg(data.content, 'assistant', false);
                    }
                };
                ws.onclose = () => {
                    status.innerText = 'Disconnected - Reconnecting...';
                    infoStatus.innerText = 'Disconnected';
                    infoStatus.classList.add('disconnected');
                    setTimeout(connect, 2000);
                };
                ws.onerror = (e) => { status.innerText = 'Error: ' + e; };
            }

            function setMode(newMode) {
                mode = newMode;
                messageCount = 0;
                infoMessages.innerText = '0';
                document.querySelectorAll('.sidebar button').forEach(b => b.classList.remove('active'));
                document.getElementById('btn-' + newMode).classList.add('active');
                chat.innerHTML = '';
                if (ws) ws.close();
                connect();
            }

            function appendMsg(text, role, streaming) {
                let last = chat.lastElementChild;
                if (streaming && last && last.dataset.streaming === 'true') {
                    last.innerHTML += text;
                    last.scrollIntoView({ behavior: 'smooth' });
                    return;
                }
                const div = document.createElement('div');
                div.className = 'message ' + role;
                div.innerHTML = text;
                div.dataset.streaming = streaming ? 'true' : 'false';
                chat.appendChild(div);
                div.scrollIntoView({ behavior: 'smooth' });
            }

            function renderTools(tools) {
                if (!tools || tools.length === 0) {
                    infoTools.innerHTML = '<div class="tool-item"><div class="tool-desc">No tools</div></div>';
                    return;
                }
                infoTools.innerHTML = tools.map(t =>
                    `<div class="tool-item">
                        <div class="tool-name">${escapeHtml(t.name)}</div>
                        <div class="tool-desc">${escapeHtml(t.description)}</div>
                    </div>`
                ).join('');
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            function send() {
                const input = document.getElementById('msgInput');
                const text = input.value.trim();
                if (!text || !ws) return;
                appendMsg(text, 'user', false);
                messageCount++;
                infoMessages.innerText = messageCount;
                ws.send(JSON.stringify({ message: text }));
                input.value = '';
            }

            connect();
        </script>
    </body>
    </html>
    """


@app.websocket("/ws/{agent_type}")
async def websocket_endpoint(websocket: WebSocket, agent_type: str) -> None:
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()

    # Get or create agent
    agent_key = f"ws_{agent_type}_{id(websocket)}"
    if agent_key not in _agent_instances:
        llm = OpenAIProvider()
        if agent_type == "chat":
            agent = ChatAgent(name="chat", llm=llm)
        elif agent_type == "react":
            agent = ReActAgent(name="react", llm=llm, tool_registry=tool_registry)
        elif agent_type == "rag":
            agent = RAGAgent(name="rag", llm=llm)
        else:
            agent = ChatAgent(name="chat", llm=llm)
        _agent_instances[agent_key] = agent

    agent = _agent_instances[agent_key]

    # Send initial config with model and tools info
    tools_data = [
        {"name": tool.name, "description": tool.description}
        for tool in tool_registry.list_tools()
    ] if agent_type == "react" else []
    await websocket.send_json({
        "type": "config",
        "model": agent.llm.default_model,
        "tools": tools_data,
    })

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                continue

            with tracer.start_trace("websocket_chat", agent=agent_type):
                response = await agent.run(user_message)

            await websocket.send_json({
                "type": "message",
                "content": response.content,
                "model": response.model,
            })

    except WebSocketDisconnect:
        # Cleanup
        _agent_instances.pop(agent_key, None)
    except Exception as e:
        await websocket.send_json({"type": "error", "content": str(e)})
        _agent_instances.pop(agent_key, None)


@app.post("/api/ingest")
async def ingest_document(file_path: str) -> dict[str, Any]:
    """Ingest a document for RAG."""
    from lcode.rag.vector_store import VectorStore
    from lcode.rag.loader import DocumentLoader

    docs = DocumentLoader.load_file(file_path)
    store = VectorStore()
    store.add_documents(docs)

    return {"status": "ok", "chunks": len(docs), "total": store.count()}


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": settings.app_name}
