"""Plugin system for dynamic tool and skill loading."""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from lcode.core.config import settings
from lcode.tools.registry import ToolRegistry, tool_registry


class PluginLoader:
    """Dynamic plugin loader.

    Scans the plugin directory and loads Python modules that
    register tools or skills.
    """

    def __init__(self, plugin_dir: Path | None = None, registry: ToolRegistry | None = None) -> None:
        self.plugin_dir = plugin_dir or settings.plugin_dir
        self.registry = registry or tool_registry
        self.loaded_plugins: list[str] = []

    def discover(self) -> list[Path]:
        """Discover all plugin files in the plugin directory."""
        if not self.plugin_dir.exists():
            return []

        plugins = []
        for file_path in self.plugin_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            plugins.append(file_path)
        return plugins

    def load(self, file_path: Path) -> bool:
        """Load a single plugin file.

        Args:
            file_path: Path to the Python plugin file.

        Returns:
            True if loaded successfully.
        """
        module_name = f"lcode.plugins.dynamic.{file_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            return False

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Inject the registry into the module namespace
        module.__dict__["registry"] = self.registry
        module.__dict__["register_tool"] = self.registry.register

        try:
            spec.loader.exec_module(module)
            self.loaded_plugins.append(file_path.stem)
            return True
        except Exception as e:
            print(f"Failed to load plugin {file_path}: {e}")
            return False

    def load_all(self) -> dict[str, bool]:
        """Load all discovered plugins.

        Returns:
            Dict mapping plugin name to success status.
        """
        results = {}
        for plugin in self.discover():
            results[plugin.stem] = self.load(plugin)
        return results

    def get_loaded(self) -> list[str]:
        """Return list of loaded plugin names."""
        return self.loaded_plugins.copy()


class SkillSystem:
    """Skills are higher-level capabilities composed of multiple tools.

    A skill can be thought of as a reusable workflow or pattern.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Any] = {}

    def register_skill(self, name: str, skill: Any) -> None:
        """Register a skill.

        Args:
            name: Skill identifier.
            skill: Skill object or callable.
        """
        self._skills[name] = skill

    def get_skill(self, name: str) -> Any | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())

    async def execute_skill(self, name: str, **kwargs: Any) -> Any:
        """Execute a skill.

        Args:
            name: Skill name.
            **kwargs: Arguments passed to the skill.

        Returns:
            Skill execution result.
        """
        skill = self.get_skill(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found.")

        if inspect.iscoroutinefunction(skill):
            return await skill(**kwargs)
        elif callable(skill):
            return skill(**kwargs)
        elif hasattr(skill, "run"):
            if inspect.iscoroutinefunction(skill.run):
                return await skill.run(**kwargs)
            return skill.run(**kwargs)
        else:
            raise ValueError(f"Skill '{name}' is not executable.")


# Global instances
plugin_loader = PluginLoader()
skill_system = SkillSystem()
