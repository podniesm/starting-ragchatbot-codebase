"""Shared pytest fixtures for RAG chatbot tests"""

import pytest
import sys
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore


@dataclass
class TestConfig:
    """Test configuration with corrected MAX_RESULTS"""

    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5  # FIXED: production config has 0
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration with temporary ChromaDB path"""
    config = TestConfig()
    config.CHROMA_PATH = str(tmp_path / "test_chroma")
    return config


@pytest.fixture
def test_vector_store(tmp_path):
    """Create VectorStore with temporary ChromaDB for testing"""
    chroma_path = str(tmp_path / "test_chroma")
    return VectorStore(
        chroma_path=chroma_path,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5,  # Override buggy config value
    )


@pytest.fixture
def sample_course():
    """Create sample Course object for testing"""
    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"
            ),
            Lesson(lesson_number=1, title="Basics", lesson_link="https://example.com/lesson1"),
        ],
    )


@pytest.fixture
def sample_chunks():
    """Create sample CourseChunk objects for testing"""
    return [
        CourseChunk(
            content="This is test content about Python programming language.",
            course_title="Test Course",
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="This is content about machine learning and AI.",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def mock_anthropic_response():
    """Create mock Anthropic API response for direct answer"""
    response = MagicMock()
    response.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.text = "This is a direct answer from Claude."
    response.content = [text_block]
    return response


@pytest.fixture
def mock_tool_use_response():
    """Create mock Anthropic API response for tool use"""
    response = MagicMock()
    response.stop_reason = "tool_use"

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "tool_123"
    tool_block.input = {"query": "test query"}

    response.content = [tool_block]
    return response


# =============================================================================
# API Testing Fixtures
# =============================================================================


@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem for API testing"""
    mock = MagicMock()

    # Mock session manager
    mock.session_manager = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-123"

    # Mock query method
    mock.query.return_value = (
        "This is a test answer about Python.",
        [{"title": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}],
    )

    # Mock get_course_analytics
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Basics", "Machine Learning"],
    }

    return mock


@pytest.fixture
def mock_rag_system_error():
    """Create a mock RAGSystem that raises errors for testing error handling"""
    mock = MagicMock()
    mock.session_manager = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-456"
    mock.query.side_effect = Exception("RAG system error")
    mock.get_course_analytics.side_effect = Exception("Analytics error")
    return mock


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {"query": "What is Python?", "session_id": None}


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request data with session ID"""
    return {"query": "Tell me more about that topic", "session_id": "existing-session-789"}
