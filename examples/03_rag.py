"""Example: RAG Agent with document ingestion."""

import asyncio

from lcode.agents.rag_agent import RAGAgent
from lcode.llm.openai_provider import OpenAIProvider


async def main() -> None:
    llm = OpenAIProvider()
    agent = RAGAgent(name="rag_demo", llm=llm)

    # Ingest a sample document
    sample_doc = "./data/sample.md"
    import os
    os.makedirs("./data", exist_ok=True)
    with open(sample_doc, "w", encoding="utf-8") as f:
        f.write("""
# LCode Framework

LCode is an AI Agent Framework written in Python.

## Features

- Single Agent Chat
- Tool Calling with ReAct
- RAG with ChromaDB
- Multi-Agent Collaboration
- Web UI with FastAPI

## Architecture

The framework uses a modular design with clear separation of concerns.
""")

    print("Ingesting document...")
    count = await agent.ingest(sample_doc)
    print(f"Ingested {count} chunks")

    # Ask questions about the document
    questions = [
        "What is LCode?",
        "What features does LCode have?",
        "What database does LCode use for RAG?",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        response = await agent.run(q)
        print(f"Agent: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
