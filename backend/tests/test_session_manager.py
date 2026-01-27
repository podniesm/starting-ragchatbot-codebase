"""Tests for SessionManager"""
import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from session_manager import SessionManager, Message


class TestSessionCreation:
    """Tests for session creation"""

    def test_create_session_returns_id(self):
        """Test that create_session returns a session ID"""
        manager = SessionManager(max_history=5)

        session_id = manager.create_session()

        assert session_id is not None
        assert isinstance(session_id, str)
        assert session_id.startswith("session_")

    def test_create_session_increments_counter(self):
        """Test that session IDs increment"""
        manager = SessionManager(max_history=5)

        id1 = manager.create_session()
        id2 = manager.create_session()

        assert id1 == "session_1"
        assert id2 == "session_2"

    def test_create_session_adds_to_sessions(self):
        """Test that created session is stored"""
        manager = SessionManager()

        session_id = manager.create_session()

        assert session_id in manager.sessions
        assert manager.sessions[session_id] == []


class TestMessageHandling:
    """Tests for message handling"""

    def test_add_message(self):
        """Test adding a single message"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_message(session_id, "user", "Hello")

        assert len(manager.sessions[session_id]) == 1
        assert manager.sessions[session_id][0].role == "user"
        assert manager.sessions[session_id][0].content == "Hello"

    def test_add_message_creates_session_if_missing(self):
        """Test that add_message creates session if it doesn't exist"""
        manager = SessionManager()

        manager.add_message("new_session", "user", "Hello")

        assert "new_session" in manager.sessions
        assert len(manager.sessions["new_session"]) == 1

    def test_add_exchange(self):
        """Test adding a complete Q&A exchange"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_exchange(session_id, "Question?", "Answer!")

        messages = manager.sessions[session_id]
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Question?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Answer!"


class TestConversationHistory:
    """Tests for conversation history retrieval"""

    def test_get_history_formats_messages(self):
        """Test that history is formatted correctly"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_exchange(session_id, "Q1", "A1")

        history = manager.get_conversation_history(session_id)

        assert "User: Q1" in history
        assert "Assistant: A1" in history

    def test_get_history_for_empty_session(self):
        """Test getting history for session with no messages"""
        manager = SessionManager()
        session_id = manager.create_session()

        history = manager.get_conversation_history(session_id)

        assert history is None

    def test_get_history_for_nonexistent_session(self):
        """Test getting history for non-existent session"""
        manager = SessionManager()

        history = manager.get_conversation_history("nonexistent")

        assert history is None

    def test_get_history_with_none_session_id(self):
        """Test getting history with None session_id"""
        manager = SessionManager()

        history = manager.get_conversation_history(None)

        assert history is None

    def test_get_history_multiple_exchanges(self):
        """Test history with multiple exchanges"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_exchange(session_id, "Q1", "A1")
        manager.add_exchange(session_id, "Q2", "A2")

        history = manager.get_conversation_history(session_id)

        assert "User: Q1" in history
        assert "Assistant: A1" in history
        assert "User: Q2" in history
        assert "Assistant: A2" in history


class TestHistoryLimiting:
    """Tests for history size limiting"""

    def test_history_limit_enforced(self):
        """Test that history is trimmed to max_history"""
        manager = SessionManager(max_history=2)
        session_id = manager.create_session()

        # Add more exchanges than max_history
        for i in range(5):
            manager.add_exchange(session_id, f"Q{i}", f"A{i}")

        # Should only keep last max_history*2 messages (4 messages = 2 exchanges)
        assert len(manager.sessions[session_id]) == 4

    def test_most_recent_messages_kept(self):
        """Test that most recent messages are kept when limiting"""
        manager = SessionManager(max_history=2)
        session_id = manager.create_session()

        for i in range(5):
            manager.add_exchange(session_id, f"Q{i}", f"A{i}")

        history = manager.get_conversation_history(session_id)

        # Should have Q3, A3, Q4, A4 (most recent)
        assert "Q3" in history
        assert "A4" in history
        # Should NOT have older messages
        assert "Q0" not in history
        assert "Q1" not in history


class TestClearSession:
    """Tests for clearing sessions"""

    def test_clear_session(self):
        """Test clearing session history"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_exchange(session_id, "Q", "A")
        assert len(manager.sessions[session_id]) == 2

        manager.clear_session(session_id)

        assert manager.sessions[session_id] == []

    def test_clear_nonexistent_session_no_error(self):
        """Test that clearing nonexistent session doesn't error"""
        manager = SessionManager()

        # Should not raise
        manager.clear_session("nonexistent")


class TestMessageDataclass:
    """Tests for Message dataclass"""

    def test_message_creation(self):
        """Test creating a Message"""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_equality(self):
        """Test Message equality"""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")

        assert msg1 == msg2
