"""RAG Agent for document Q&A."""

from typing import Any

from lcode.agents.base import BaseAgent
from lcode.llm.base import LLMResponse, Message
from lcode.rag.loader import DocumentLoader
from lcode.rag.vector_store import VectorStore


class RAGAgent(BaseAgent):
    """An agent with Retrieval-Augmented Generation capabilities.

    This is the Level 3 implementation.
    It can load documents, store them in a vector database,
    and answer questions based on retrieved context.
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        vector_store: VectorStore | None = None,
        system_prompt: str = "You are a helpful assistant with access to a knowledge base.",
    ) -> None:
        super().__init__(name, llm, system_prompt)
        self.vector_store = vector_store or VectorStore()

    async def ingest(self, file_path: str) -> int:
        """Ingest documents from a file into the vector store.

        Args:
            file_path: Path to document file.

        Returns:
            Number of chunks ingested.
        """
        documents = DocumentLoader.load_file(file_path)
        self.vector_store.add_documents(documents)
        return len(documents)

    async def ingest_directory(self, dir_path: str, pattern: str = "*") -> int:
        """Ingest all matching files from a directory.

        Args:
            dir_path: Directory path.
            pattern: Glob pattern.

        Returns:
            Total number of chunks ingested.
        """
        documents = DocumentLoader.load_directory(dir_path, pattern)
        if documents:
            self.vector_store.add_documents(documents)
        return len(documents)

    async def run(self, user_input: str, **kwargs: Any) -> LLMResponse:
        """Answer a question using RAG.

        1. Retrieve relevant documents
        2. Build context-enhanced prompt
        3. Generate answer
        """
        # Retrieve relevant documents
        results = self.vector_store.query(user_input, top_k=kwargs.get("top_k", 5))

        if not results:
            # No documents found - answer without context
            messages = self._build_messages(user_input)
            response = await self._call_llm(messages, **kwargs)
            self.add_to_history("user", user_input)
            self.add_to_history("assistant", response.content)
            return response

        # Build context from retrieved documents
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}]\n{result['content']}")
        context = "\n\n".join(context_parts)

        # Build RAG-enhanced prompt
        rag_prompt = (
            f"{user_input}\n\n"
            f"Use the following retrieved documents to answer the question:\n\n"
            f"{context}\n\n"
            "If the documents don't contain the answer, say so clearly."
        )

        messages = self._build_messages(rag_prompt)
        response = await self._call_llm(messages, **kwargs)

        self.add_to_history("user", user_input)
        self.add_to_history("assistant", response.content)

        return response
