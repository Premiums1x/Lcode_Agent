"""Example: Simple chat with ChatAgent."""

import asyncio

from lcode.agents.chat_agent import ChatAgent
from lcode.llm.openai_provider import OpenAIProvider


async def main() -> None:
    llm = OpenAIProvider()
    agent = ChatAgent(name="example", llm=llm)

    questions = [
        "Hello! What can you do?",
        "What's the weather like? (This is just a test)",
    ]

    for q in questions:
        print(f"User: {q}")
        response = await agent.run(q)
        print(f"Agent: {response.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
