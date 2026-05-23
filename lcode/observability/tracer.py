"""Structured logging and tracing for LCode.

This is part of the Level 5 production-grade requirements.
"""

import asyncio
import functools
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

import structlog
from loguru import logger as loguru_logger

# Context variables for trace propagation
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)


# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


class Tracer:
    """Simple tracer for tracking agent execution flows."""

    def __init__(self, service_name: str = "lcode") -> None:
        self.service_name = service_name
        self.logger = structlog.get_logger(service_name)

    def start_trace(self, name: str, **attributes: Any) -> "TraceSpan":
        """Start a new trace/span.

        Args:
            name: Operation name.
            **attributes: Additional attributes.

        Returns:
            TraceSpan context manager.
        """
        return TraceSpan(self, name, **attributes)

    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log an event with current trace context."""
        trace_id = _trace_id.get()
        span_id = _span_id.get()
        self.logger.info(
            event,
            trace_id=trace_id,
            span_id=span_id,
            **kwargs,
        )

    def log_agent_action(
        self,
        agent_name: str,
        action: str,
        input_data: str | None = None,
        output_data: str | None = None,
        duration_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an agent action with full context."""
        self.logger.info(
            "agent_action",
            agent=agent_name,
            action=action,
            input=input_data,
            output=output_data,
            duration_ms=duration_ms,
            trace_id=_trace_id.get(),
            **kwargs,
        )


class TraceSpan:
    """A trace span context manager."""

    def __init__(self, tracer: Tracer, name: str, **attributes: Any) -> None:
        self.tracer = tracer
        self.name = name
        self.attributes = attributes
        self.start_time: float = 0.0
        self.trace_id: str | None = None
        self.span_id: str | None = None
        self._token_trace: Any = None
        self._token_span: Any = None

    def __enter__(self) -> "TraceSpan":
        self.start_time = time.time()
        self.trace_id = _trace_id.get() or str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())

        self._token_trace = _trace_id.set(self.trace_id)
        self._token_span = _span_id.set(self.span_id)

        self.tracer.logger.info(
            "span_start",
            name=self.name,
            trace_id=self.trace_id,
            span_id=self.span_id,
            **self.attributes,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = (time.time() - self.start_time) * 1000
        self.tracer.logger.info(
            "span_end",
            name=self.name,
            trace_id=self.trace_id,
            span_id=self.span_id,
            duration_ms=duration,
            error=exc_type is not None,
            **self.attributes,
        )
        if self._token_trace:
            _trace_id.reset(self._token_trace)
        if self._token_span:
            _span_id.reset(self._token_span)

    async def __aenter__(self) -> "TraceSpan":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


# Global tracer instance
tracer = Tracer()


def trace_method(name: str | None = None) -> Callable[..., Any]:
    """Decorator to trace a method."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or func.__qualname__
            with tracer.start_trace(span_name, function=func.__qualname__):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            span_name = name or func.__qualname__
            with tracer.start_trace(span_name, function=func.__qualname__):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Re-export loguru logger for convenience
logger = loguru_logger
