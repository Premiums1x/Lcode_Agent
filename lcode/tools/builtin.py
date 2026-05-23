"""Built-in tools for LCode Agent Framework."""

import asyncio
import json
import math

from lcode.tools.registry import tool_registry


@tool_registry.register(
    name="calculator",
    description="Evaluate a mathematical expression. Supports basic arithmetic, powers, sqrt, and common math functions.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g., '2 + 2 * 5' or 'sqrt(16)'",
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Whitelist of safe functions and constants
        safe_dict = {
            "abs": abs,
            "float": float,
            "int": int,
            "max": max,
            "min": min,
            "pow": pow,
            "round": round,
            "sum": sum,
            "math": math,
            "__builtins__": {},
        }
        # Add math module functions
        for attr in dir(math):
            if not attr.startswith("_"):
                safe_dict[attr] = getattr(math, attr)

        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool_registry.register(
    name="web_search",
    description="Search the web for information. Returns a summary of search results.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using DuckDuckGo (no API key required).

    Note: This is a simplified implementation. In production,
    you might want to use a proper search API or scraping.
    """
    import urllib.parse
    import urllib.request
    from html.parser import HTMLParser

    num_results = min(num_results, 10)

    try:
        # Use DuckDuckGo HTML version (no JS required)
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        req = urllib.request.Request(url, headers=headers)

        # Run blocking I/O in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, urllib.request.urlopen, req)
        html = await loop.run_in_executor(None, response.read)
        html_str = html.decode("utf-8", errors="ignore")

        # Simple HTML parsing to extract results
        results = []
        current = {"title": "", "snippet": "", "url": ""}
        in_result = False
        in_title = False
        in_snippet = False

        class DDGParser(HTMLParser):
            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                nonlocal in_result, in_title, in_snippet
                attrs_dict = dict(attrs)
                cls: str = str(attrs_dict.get("class", "") or "")

                if "result" in cls and tag == "div":
                    in_result = True
                if in_result and tag == "a" and "result__a" in cls:
                    in_title = True
                    current["url"] = str(attrs_dict.get("href", "") or "")
                if in_result and tag == "a" and "result__snippet" in cls:
                    in_snippet = True

            def handle_endtag(self, tag: str) -> None:
                nonlocal in_result, in_title, in_snippet
                if in_title and tag == "a":
                    in_title = False
                if in_snippet and tag == "a":
                    in_snippet = False
                    in_result = False
                    results.append(dict(current))
                    current["title"] = ""
                    current["snippet"] = ""
                    current["url"] = ""

            def handle_data(self, data: str) -> None:
                if in_title:
                    current["title"] += data
                if in_snippet:
                    current["snippet"] += data

        parser = DDGParser()
        parser.feed(html_str)

        if not results:
            return f"No results found for '{query}'."

        output = []
        for i, r in enumerate(results[:num_results], 1):
            title = r.get("title", "").strip()
            snippet = r.get("snippet", "").strip()
            url = r.get("url", "").strip()
            output.append(f"{i}. {title}\n   {snippet}\n   URL: {url}")

        return "\n\n".join(output)

    except Exception as e:
        return f"Search failed: {e}"


@tool_registry.register(
    name="python_executor",
    description="Execute Python code safely and return the output. Useful for data processing, file operations, and complex calculations.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            }
        },
        "required": ["code"],
    },
)
def python_executor(code: str) -> str:
    """Execute Python code in a restricted environment.

    WARNING: This runs code locally. In production, use a sandbox.
    """
    import io
    import sys
    import traceback

    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer

    try:
        # Restricted globals
        safe_globals = {
            "__builtins__": __builtins__,
            "json": json,
            "math": math,
            "print": print,
        }

        exec(code, safe_globals)

        output = stdout_buffer.getvalue()
        error = stderr_buffer.getvalue()

        if error:
            return f"Output:\n{output}\n\nError:\n{error}"
        return f"Output:\n{output}" if output else "Code executed successfully with no output."

    except Exception:
        error = traceback.format_exc()
        return f"Exception occurred:\n{error}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
