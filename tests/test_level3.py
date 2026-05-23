"""Tests for Level 3: Memory + RAG."""

from lcode.memory.conversation import InMemoryMemory, SQLiteMemory
from lcode.rag.loader import Document, DocumentLoader


class TestInMemoryMemory:
    """Test in-memory conversation memory."""

    def test_add_and_retrieve(self) -> None:
        """Test adding and retrieving messages."""
        memory = InMemoryMemory()
        memory.add("user", "Hello")
        memory.add("assistant", "Hi there!")

        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_limit(self) -> None:
        """Test message retrieval with limit."""
        memory = InMemoryMemory()
        for i in range(10):
            memory.add("user", f"msg{i}")

        messages = memory.get_messages(limit=5)
        assert len(messages) == 5
        assert messages[-1]["content"] == "msg9"

    def test_clear(self) -> None:
        """Test clearing memory."""
        memory = InMemoryMemory()
        memory.add("user", "Hello")
        memory.clear()

        assert len(memory.get_messages()) == 0

    def test_max_messages(self) -> None:
        """Test that old messages are dropped when max is reached."""
        memory = InMemoryMemory(max_messages=3)
        memory.add("user", "1")
        memory.add("user", "2")
        memory.add("user", "3")
        memory.add("user", "4")

        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "2"


class TestSQLiteMemory:
    """Test SQLite conversation memory."""

    def test_persistence(self, tmp_path) -> None:
        """Test that messages persist to SQLite."""
        db_path = tmp_path / "test.db"
        memory = SQLiteMemory(db_path=db_path, session_id="test_session")

        memory.add("user", "Hello")
        memory.add("assistant", "Hi!")

        messages = memory.get_messages()
        assert len(messages) == 2

        # Create new instance with same DB
        memory2 = SQLiteMemory(db_path=db_path, session_id="test_session")
        messages2 = memory2.get_messages()
        assert len(messages2) == 2

    def test_session_isolation(self, tmp_path) -> None:
        """Test that different sessions are isolated."""
        db_path = tmp_path / "test.db"
        memory1 = SQLiteMemory(db_path=db_path, session_id="session1")
        memory2 = SQLiteMemory(db_path=db_path, session_id="session2")

        memory1.add("user", "Hello from 1")
        memory2.add("user", "Hello from 2")

        assert len(memory1.get_messages()) == 1
        assert memory1.get_messages()[0]["content"] == "Hello from 1"
        assert len(memory2.get_messages()) == 1
        assert memory2.get_messages()[0]["content"] == "Hello from 2"


class TestDocumentLoader:
    """Test document loading and chunking."""

    def test_load_text_file(self, tmp_path) -> None:
        """Test loading a plain text file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("This is a test document. " * 10)

        docs = DocumentLoader.load_file(file_path)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_load_markdown(self, tmp_path) -> None:
        """Test loading a markdown file."""
        file_path = tmp_path / "test.md"
        file_path.write_text("# Title\n\n## Section 1\nContent 1\n\n## Section 2\nContent 2")

        docs = DocumentLoader.load_file(file_path)
        assert len(docs) >= 2  # At least 2 sections

    def test_split_text(self) -> None:
        """Test text splitting into chunks."""
        text = "a" * 1000
        docs = DocumentLoader._split_text(text, "test", chunk_size=100, overlap=10)

        assert len(docs) > 1
        # Check overlap
        assert docs[0].content[-10:] == docs[1].content[:10]

    def test_load_directory(self, tmp_path) -> None:
        """Test loading multiple files from directory."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        (tmp_path / "ignore.py").write_text("# Python")

        docs = DocumentLoader.load_directory(tmp_path, "*.txt")
        assert len(docs) == 2
