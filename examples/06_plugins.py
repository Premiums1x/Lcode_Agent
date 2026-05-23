"""Example: Plugin and Skills system usage."""

import asyncio

from lcode.plugins.loader import plugin_loader, skill_system
from lcode.tools.registry import tool_registry


async def main() -> None:
    # Discover and load all plugins
    print("Discovering plugins...")
    plugins = plugin_loader.discover()
    print(f"Found {len(plugins)} plugin(s)")

    for plugin_path in plugins:
        success = plugin_loader.load(plugin_path)
        status = "loaded" if success else "failed"
        print(f"  {plugin_path.name}: {status}")

    # List loaded plugins
    print(f"\nLoaded plugins: {plugin_loader.get_loaded()}")

    # List all registered tools (including from plugins)
    print("\nAll registered tools:")
    for tool in tool_registry.list_tools():
        print(f"  - {tool.name}: {tool.description}")

    # Register a custom skill
    def data_analysis_skill(query: str) -> str:
        """Analyze data based on a query."""
        return f"Analyzed: {query}"

    skill_system.register_skill("data_analysis", data_analysis_skill)
    print(f"\nRegistered skills: {skill_system.list_skills()}")

    # Execute skill
    result = await skill_system.execute_skill("data_analysis", query="sales Q3 2024")
    print(f"Skill result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
