"""Conversation memory implementations."""

import json
import sqlite3
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

from lcode.core.config import settings


class BaseMemory(ABC):
    """Abstract base class for conversation memory."""

    @abstractmethod
    def add(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to memory."""
        ...

    @abstractmethod
    def get_messages(self, limit: int | None = None) -> list[dict[str, str]]:
        """Retrieve messages from memory."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        ...


class InMemoryMemory(BaseMemory):
    """Simple in-memory conversation buffer."""

    def __init__(self, max_messages: int = 100) -> None:
        self.messages: deque[dict[str, str]] = deque(maxlen=max_messages)

    def add(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append({"role": role, "content": content})

    def get_messages(self, limit: int | None = None) -> list[dict[str, str]]:
        msgs = list(self.messages)
        if limit:
            return msgs[-limit:]
        return msgs

    def clear(self) -> None:
        self.messages.clear()


class SQLiteMemory(BaseMemory):
    """Persistent conversation memory using SQLite."""

    def __init__(self, db_path: Path | None = None, session_id: str = "default") -> None:
        self.db_path = db_path or settings.memory_db_path
        self.session_id = session_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, created_at)"
            )
            conn.commit()

    def add(self, role: str, content: str, **kwargs: Any) -> None:
        metadata = json.dumps(kwargs) if kwargs else None
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (self.session_id, role, content, metadata),
            )
            conn.commit()

    def get_messages(self, limit: int | None = None) -> list[dict[str, str]]:
        query = (
            "SELECT role, content, metadata FROM messages "
            "WHERE session_id = ? ORDER BY created_at ASC"
        )
        params: list[Any] = [self.session_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]

    def clear(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
            conn.commit()


def create_memory(
    memory_type: str | None = None,
    session_id: str = "default",
    **kwargs: Any,
) -> BaseMemory:
    """Factory function to create memory backend.

    Args:
        memory_type: 'in_memory', 'sqlite', or 'redis'. Defaults to settings.
        session_id: Conversation session identifier.
        **kwargs: Extra arguments for memory backend.

    Returns:
        Memory instance.
    """
    mem_type = memory_type or settings.memory_type

    if mem_type == "in_memory":
        return InMemoryMemory(**kwargs)
    elif mem_type == "sqlite":
        return SQLiteMemory(session_id=session_id, **kwargs)
    elif mem_type == "redis":
        raise NotImplementedError("Redis memory not yet implemented.")
    else:
        raise ValueError(f"Unknown memory type: {mem_type}")
