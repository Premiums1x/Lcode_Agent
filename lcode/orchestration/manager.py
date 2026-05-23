"""Multi-agent orchestration with Plan-and-Solve paradigm.

This is the Level 4 implementation.
It supports:
- Multi-agent collaboration
- Agent communication via Event Bus
- Task decomposition and delegation
- Agent Manager for dynamic scheduling
"""

import json
from typing import Any

from lcode.agents.base import BaseAgent
from lcode.core.events import event_bus
from lcode.llm.base import BaseLLMProvider, LLMResponse, Message


class Task:
    """Represents a decomposed task."""

    def __init__(self, task_id: str, description: str, assignee: str | None = None) -> None:
        self.task_id = task_id
        self.description = description
        self.assignee = assignee
        self.status: str = "pending"  # pending, in_progress, completed, failed
        self.result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "assignee": self.assignee,
            "status": self.status,
            "result": self.result,
        }


class AgentManager:
    """Manages a pool of agents and schedules tasks.

    Acts as the orchestrator in Plan-and-Solve:
    1. Receives a complex task
    2. Plans: Decomposes into sub-tasks
    3. Solves: Delegates to specialized agents
    4. Integrates: Combines results into final answer
    """

    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm
        self.agents: dict[str, BaseAgent] = {}
        self.tasks: list[Task] = []

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the manager."""
        self.agents[agent.name] = agent

    def unregister_agent(self, name: str) -> None:
        """Remove an agent from the manager."""
        self.agents.pop(name, None)

    async def execute(self, task_description: str, **kwargs: Any) -> LLMResponse:
        """Execute a complex task using Plan-and-Solve.

        Steps:
        1. Plan: Use LLM to decompose task
        2. Delegate: Assign sub-tasks to agents
        3. Solve: Execute sub-tasks in parallel or sequence
        4. Integrate: Combine results into final answer
        """
        # Step 1: Plan - Decompose into sub-tasks
        plan = await self._plan(task_description)
        self.tasks = [Task(t["id"], t["description"], t.get("agent")) for t in plan["tasks"]]

        # Step 2 & 3: Delegate and Solve
        for task in self.tasks:
            task.status = "in_progress"
            agent = self._select_agent(task)
            if agent:
                try:
                    response = await agent.run(task.description, **kwargs)
                    task.result = response.content
                    task.status = "completed"
                    # Publish completion event
                    await event_bus.publish(
                        "task_completed",
                        task_id=task.task_id,
                        result=task.result,
                    )
                except Exception as e:
                    task.result = f"Error: {e}"
                    task.status = "failed"
            else:
                task.status = "failed"
                task.result = f"No suitable agent found for task: {task.description}"

        # Step 4: Integrate results
        final_answer = await self._integrate(task_description)
        return LLMResponse(content=final_answer, model="orchestrator")

    async def _plan(self, task_description: str) -> dict[str, Any]:
        """Use LLM to decompose task into sub-tasks."""
        system_prompt = (
            "You are a task planner. Decompose the given task into clear, "
            "atomic sub-tasks. Each sub-task should specify which type of "
            "agent is best suited to handle it."
        )

        available_agents = ", ".join(self.agents.keys()) if self.agents else "any"

        prompt = (
            f"Available agents: {available_agents}\n\n"
            f"Task: {task_description}\n\n"
            "Respond with a JSON object containing a 'tasks' array. "
            "Each task should have: 'id', 'description', and 'agent' (agent name or 'any').\n\n"
            "Example:\n"
            '{"tasks": [{"id": "1", "description": "Search for recent news", "agent": "researcher"}]}'
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt),
        ]

        response = await self.llm.chat(messages=messages)
        content = response.content.strip()

        # Try to extract JSON
        try:
            # Find JSON in markdown code blocks
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content

            parsed: dict[str, Any] = json.loads(json_str)
            return parsed
        except json.JSONDecodeError:
            # Fallback: single task
            return {"tasks": [{"id": "1", "description": task_description, "agent": "any"}]}

    def _select_agent(self, task: Task) -> BaseAgent | None:
        """Select the best agent for a task."""
        if task.assignee and task.assignee in self.agents:
            return self.agents[task.assignee]

        # Fallback to first available agent
        if self.agents:
            return next(iter(self.agents.values()))
        return None

    async def _integrate(self, original_task: str) -> str:
        """Combine sub-task results into a coherent final answer."""
        results = []
        for task in self.tasks:
            results.append(f"Task {task.task_id}: {task.description}\nResult: {task.result}")

        results_text = "\n\n".join(results)

        prompt = (
            f"Original task: {original_task}\n\n"
            f"Sub-task results:\n\n{results_text}\n\n"
            "Please synthesize these results into a single, coherent final answer."
        )

        messages = [
            Message(
                role="system",
                content="You are an integration specialist. Combine multiple results into a clear, comprehensive answer.",
            ),
            Message(role="user", content=prompt),
        ]

        response = await self.llm.chat(messages=messages)
        return response.content
