"""Document loader for various file formats."""

from pathlib import Path
from typing import Any


class Document:
    """Represents a document chunk."""

    def __init__(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.content = content
        self.metadata = metadata or {}


class DocumentLoader:
    """Load documents from files."""

    @staticmethod
    def load_file(file_path: str | Path) -> list[Document]:
        """Load a single file and return documents.

        Supports .txt, .md, .py, .json, .csv
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8")
        ext = path.suffix.lower()

        if ext in {".md", ".markdown"}:
            return DocumentLoader._split_markdown(text, str(path))
        elif ext == ".json":
            import json

            data = json.loads(text)
            if isinstance(data, list):
                return [
                    Document(content=json.dumps(item), metadata={"source": str(path), "index": i})
                    for i, item in enumerate(data)
                ]
            return [Document(content=text, metadata={"source": str(path)})]
        elif ext == ".csv":
            return DocumentLoader._split_csv(text, str(path))
        else:
            # Plain text - split into chunks
            return DocumentLoader._split_text(text, str(path))

    @staticmethod
    def load_directory(dir_path: str | Path, pattern: str = "*") -> list[Document]:
        """Load all matching files from a directory."""
        path = Path(dir_path)
        docs = []
        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    docs.extend(DocumentLoader.load_file(file_path))
                except Exception:
                    continue
        return docs

    @staticmethod
    def _split_text(
        text: str, source: str, chunk_size: int = 512, overlap: int = 50
    ) -> list[Document]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(
                Document(content=chunk, metadata={"source": source, "chunk_start": start})
            )
            start += chunk_size - overlap
            if start >= len(text):
                break
            # Ensure we always make progress
            if start <= 0:
                start = end
        return chunks

    @staticmethod
    def _split_markdown(text: str, source: str) -> list[Document]:
        """Split markdown by headers."""
        import re

        # Split by ## or ### headers
        sections = re.split(r"\n(?=#{1,3} )", text)
        docs = []
        for i, section in enumerate(sections):
            section = section.strip()
            if section:
                docs.append(Document(content=section, metadata={"source": source, "section": i}))
        return docs if docs else DocumentLoader._split_text(text, source)

    @staticmethod
    def _split_csv(text: str, source: str) -> list[Document]:
        """Split CSV into row documents."""
        import csv
        import io

        reader = csv.DictReader(io.StringIO(text))
        return [
            Document(content=str(row), metadata={"source": source, "row": i})
            for i, row in enumerate(reader)
        ]
