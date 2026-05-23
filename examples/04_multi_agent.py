"""Example: Multi-Agent collaboration with Plan-and-Solve."""

import asyncio

from lcode.agents.chat_agent import ChatAgent
from lcode.agents.react_agent import ReActAgent
from lcode.llm.openai_provider import OpenAIProvider
from lcode.orchestration.manager import AgentManager
from lcode.tools.builtin import calculator, python_executor, web_search  # noqa: F401
from lcode.tools.registry import tool_registry


async def main() -> None:
    llm = OpenAIProvider()

    # Create specialized agents
    researcher = ChatAgent(
        name="researcher",
        llm=llm,
        system_prompt="You are a research specialist. Gather and summarize information.",
    )
    calculator_agent = ReActAgent(
        name="calculator",
        llm=llm,
        tool_registry=tool_registry,
        system_prompt="You are a math specialist. Solve numerical problems accurately.",
    )
    coder = ChatAgent(
        name="coder",
        llm=llm,
        system_prompt="You are a Python coding specialist. Write clean, efficient code.",
    )

    # Create manager and register agents
    manager = AgentManager(llm=llm)
    manager.register_agent(researcher)
    manager.register_agent(calculator_agent)
    manager.register_agent(coder)

    # Complex task that requires collaboration
    task = (
        "Research the current state of AI agent frameworks in 2024, "
        "calculate how many months have passed since January 2023, "
        "and write a simple Python script that prints a summary."
    )

    print(f"Task: {task}\n")
    print("Executing with Plan-and-Solve...\n")

    response = await manager.execute(task)
    print(f"Final Answer:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
