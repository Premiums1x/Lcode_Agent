"""Example: ReAct Agent with tool calling."""

import asyncio

from lcode.agents.react_agent import ReActAgent
from lcode.llm.openai_provider import OpenAIProvider
from lcode.tools.builtin import calculator, python_executor, web_search  # noqa: F401
from lcode.tools.registry import tool_registry


async def main() -> None:
    llm = OpenAIProvider()
    agent = ReActAgent(
        name="react_demo",
        llm=llm,
        tool_registry=tool_registry,
        system_prompt="You are a helpful assistant with access to tools.",
    )

    questions = [
        "What is the square root of 144 plus 15?",
        "Search for recent news about AI agents",
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"User: {q}")
        print("=" * 50)
        response = await agent.run(q)
        print(f"Agent: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
