"""Command-line interface for LCode.

Level 1 requirement: Command-line runnable.
"""

import asyncio
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lcode.agents.chat_agent import ChatAgent
from lcode.agents.rag_agent import RAGAgent
from lcode.agents.react_agent import ReActAgent
from lcode.core.config import settings
from lcode.llm.openai_provider import OpenAIProvider
from lcode.observability.tracer import tracer
from lcode.plugins.loader import plugin_loader
from lcode.tools.builtin import calculator, python_executor, web_search  # noqa: F401 - registers tools
from lcode.tools.registry import Tool, tool_registry
from lcode.web.app import app as web_app

console = Console()
cli = typer.Typer(
    name="lcode",
    help="LCode - AI Agent Framework",
    rich_markup_mode="rich",
)


def _print_banner() -> None:
    """Print the CLI banner."""
    banner = """
    ██╗      ██████╗ ██████╗ ██████╗ ███████╗
    ██║     ██╔════╝██╔═══██╗██╔══██╗██╔════╝
    ██║     ██║     ██║   ██║██║  ██║█████╗
    ██║     ██║     ██║   ██║██║  ██║██╔══╝
    ███████╗╚██████╗╚██████╔╝██████╔╝███████╗
    ╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
    AI Agent Framework v0.1.0
    """
    console.print(Panel(banner, title="Welcome", border_style="cyan"))


def _print_session_info(
    agent_type: str, model: str, temperature: float, tools: list[Tool]
) -> None:
    """Print session info on the right side, once at startup."""
    # Build info content
    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column("Key", style="dim", width=10)
    info_table.add_column("Value", style="bold")
    info_table.add_row("Agent", agent_type.upper())
    info_table.add_row("Model", f"[red]{model}[/red]")
    info_table.add_row("Temp", str(temperature))

    if tools:
        info_table.add_row("", "")
        info_table.add_row("[bold]Tools[/bold]", "")
        for tool in tools:
            info_table.add_row(f"  • {tool.name}", tool.description[:40])

    # Left side: welcome text; Right side: info panel
    welcome = Text(
        "Welcome to LCode!\nStart chatting below.\n\n"
        "[dim]Type 'exit' or 'quit' to end.[/dim]",
        style="",
    )

    layout_table = Table(show_header=False, box=None, expand=True, padding=(0, 2))
    layout_table.add_column("Chat", ratio=3)
    layout_table.add_column("Info", ratio=1, min_width=30)
    layout_table.add_row(
        Panel(welcome, border_style="dim"),
        Panel(info_table, title="[bold]Session Info[/bold]", border_style="blue"),
    )
    console.print(layout_table)
    console.print()


@cli.command()
def chat(
    model: str = typer.Option(settings.default_model, "--model", "-m", help="LLM model to use"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Temperature"),
    agent_type: str = typer.Option("chat", "--agent", "-a", help="Agent type: chat, react, rag"),
) -> None:
    """Start an interactive chat session."""
    _print_banner()

    # Initialize LLM
    try:
        llm = OpenAIProvider(default_model=model)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable.[/yellow]")
        raise typer.Exit(1)

    # Create agent
    tools_list: list[Tool] = []
    if agent_type == "react":
        agent = ReActAgent(name="react", llm=llm, tool_registry=tool_registry)
        tools_list = tool_registry.list_tools()
    elif agent_type == "rag":
        agent = RAGAgent(name="rag", llm=llm)
    else:
        agent = ChatAgent(name="chat", llm=llm)

    # Print session info ONCE — right side, never re-rendered
    _print_session_info(agent_type, agent.llm.default_model, temperature, tools_list)

    # Prompt always shows key info so user never loses context
    prompt = f"[bold cyan][{agent_type.upper()} | {agent.llm.default_model}][/] You: "

    async def _run_chat() -> None:
        with tracer.start_trace("cli_chat_session", agent=agent_type):
            while True:
                try:
                    user_input = console.input(prompt)
                    user_input = user_input.strip()

                    if user_input.lower() in {"exit", "quit", "q"}:
                        console.print("[dim]Goodbye![/dim]")
                        break

                    if not user_input:
                        continue

                    with console.status("[bold green]Thinking...[/bold green]"):
                        response = await agent.run(user_input, temperature=temperature)

                    # Render response
                    if any(c in response.content for c in ["#", "*", "`", "|"]):
                        md = Markdown(response.content)
                        console.print(Panel(md, title="[bold green]LCode[/bold green]", border_style="green"))
                    else:
                        console.print(Panel(response.content, title="[bold green]LCode[/bold green]", border_style="green"))

                except KeyboardInterrupt:
                    console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

    asyncio.run(_run_chat())


@cli.command()
def server(
    host: str = typer.Option(settings.web_host, "--host", "-h", help="Host to bind"),
    port: int = typer.Option(settings.web_port, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(settings.web_reload, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the LCode Web UI server."""
    import uvicorn

    console.print(f"[green]Starting LCode server at http://{host}:{port}[/green]")
    uvicorn.run(
        "lcode.web.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    pattern: str = typer.Option("*", "--pattern", help="File pattern for directories"),
) -> None:
    """Ingest documents into the RAG vector store."""
    from pathlib import Path

    from lcode.rag.loader import DocumentLoader
    from lcode.rag.vector_store import VectorStore

    p = Path(path)
    store = VectorStore()

    async def _do_ingest() -> None:
        if p.is_file():
            docs = DocumentLoader.load_file(p)
            store.add_documents(docs)
            console.print(f"[green]Ingested {len(docs)} chunks from {path}[/green]")
        elif p.is_dir():
            docs = DocumentLoader.load_directory(p, pattern)
            if docs:
                store.add_documents(docs)
            console.print(f"[green]Ingested {len(docs)} chunks from directory {path}[/green]")
        else:
            console.print(f"[red]Path not found: {path}[/red]")
            raise typer.Exit(1)

        console.print(f"[dim]Total documents in store: {store.count()}[/dim]")

    asyncio.run(_do_ingest())


@cli.command()
def tools() -> None:
    """List all available tools."""
    console.print("[bold cyan]Available Tools:[/bold cyan]\n")
    for tool in tool_registry.list_tools():
        console.print(f"[bold]{tool.name}[/bold] - {tool.description}")
    if not tool_registry.list_tools():
        console.print("[dim]No tools registered.[/dim]")


@cli.command()
def plugins(
    load: bool = typer.Option(False, "--load", "-l", help="Auto-discover and load plugins"),
    list_loaded: bool = typer.Option(False, "--list", help="List loaded plugins"),
) -> None:
    """Manage plugins."""
    if load:
        results = plugin_loader.load_all()
        for name, success in results.items():
            status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
            console.print(f"{name}: {status}")

    if list_loaded or not load:
        loaded = plugin_loader.get_loaded()
        if loaded:
            console.print("[bold]Loaded plugins:[/bold]")
            for name in loaded:
                console.print(f"  • {name}")
        else:
            console.print("[dim]No plugins loaded.[/dim]")


@cli.command()
def config() -> None:
    """Show current configuration."""
    console.print("[bold cyan]LCode Configuration[/bold cyan]\n")
    console.print(f"Default Model: {settings.default_model}")
    console.print(f"Temperature: {settings.default_temperature}")
    console.print(f"Memory Type: {settings.memory_type}")
    console.print(f"Vector DB: {settings.vector_db_path}")
    console.print(f"Plugin Dir: {settings.plugin_dir}")
    console.print(f"Web UI: {settings.web_host}:{settings.web_port}")


def main() -> Any:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()