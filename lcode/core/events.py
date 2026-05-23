"""Event bus for decoupled communication between agents and components."""

import asyncio
from collections.abc import Callable
from typing import Any


class EventBus:
    """Simple in-memory event bus for agent communication.

    Supports both sync and async subscribers.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., Any]]] = {}
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb is not callback
            ]

    async def publish(self, event_type: str, **kwargs: Any) -> list[Any]:
        """Publish an event to all subscribers.

        Returns a list of results from subscribers.
        """
        callbacks = self._subscribers.get(event_type, [])
        results = []
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    result = await cb(**kwargs)
                else:
                    result = cb(**kwargs)
                results.append(result)
            except Exception as e:
                # Log but don't break other subscribers
                results.append(e)
        return results

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()


# Global event bus instance
event_bus = EventBus()
