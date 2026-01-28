"""Tests for RAGSystem.query() orchestration"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @patch("rag_system.AIGenerator")
    def test_query_returns_tuple(self, mock_ai_class, test_config):
        """Test that query() returns (response, sources) tuple"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Test response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        result = rag.query("What is Python?")

        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert isinstance(response, str)
        assert isinstance(sources, list)

    @patch("rag_system.AIGenerator")
    def test_query_returns_ai_response(self, mock_ai_class, test_config):
        """Test that query returns the AI generator's response"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "AI generated answer"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        response, _ = rag.query("Test question")

        assert response == "AI generated answer"

    @patch("rag_system.AIGenerator")
    def test_query_passes_tools_to_ai_generator(self, mock_ai_class, test_config):
        """Test that tools are passed to AI generator"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        rag.query("Search for MCP")

        call_args = mock_ai_instance.generate_response.call_args
        assert "tools" in call_args.kwargs
        assert "tool_manager" in call_args.kwargs
        assert len(call_args.kwargs["tools"]) >= 1  # At least search tool

    @patch("rag_system.AIGenerator")
    def test_query_passes_tool_manager(self, mock_ai_class, test_config):
        """Test that tool_manager is passed for tool execution"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        rag.query("Query")

        call_args = mock_ai_instance.generate_response.call_args
        assert call_args.kwargs["tool_manager"] is not None

    @patch("rag_system.AIGenerator")
    def test_query_creates_prompt_with_query(self, mock_ai_class, test_config):
        """Test that query is wrapped in prompt"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        rag.query("What is MCP?")

        call_args = mock_ai_instance.generate_response.call_args
        assert "What is MCP?" in call_args.kwargs["query"]


class TestRAGSystemWithSessions:
    """Tests for session handling in RAGSystem"""

    @patch("rag_system.AIGenerator")
    def test_query_works_without_session(self, mock_ai_class, test_config):
        """Test that query works when no session_id provided"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        response, sources = rag.query("Test query", session_id=None)

        assert response == "Response"

    @patch("rag_system.AIGenerator")
    def test_query_uses_conversation_history(self, mock_ai_class, test_config):
        """Test that query includes conversation history for existing session"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        # Create session with history
        session_id = rag.session_manager.create_session()
        rag.session_manager.add_exchange(session_id, "Previous Q", "Previous A")

        rag.query("Follow-up question", session_id=session_id)

        call_args = mock_ai_instance.generate_response.call_args
        history = call_args.kwargs.get("conversation_history")
        assert history is not None
        assert "Previous Q" in history

    @patch("rag_system.AIGenerator")
    def test_query_updates_session_history(self, mock_ai_class, test_config):
        """Test that query updates session history after response"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "New response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        session_id = rag.session_manager.create_session()
        rag.query("User question", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert "User question" in history
        assert "New response" in history

    @patch("rag_system.AIGenerator")
    def test_query_no_history_for_nonexistent_session(self, mock_ai_class, test_config):
        """Test that nonexistent session has no history"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        rag.query("Query", session_id="nonexistent_session")

        call_args = mock_ai_instance.generate_response.call_args
        history = call_args.kwargs.get("conversation_history")
        assert history is None


class TestRAGSystemSourceTracking:
    """Tests for source tracking in RAGSystem"""

    @patch("rag_system.AIGenerator")
    def test_query_returns_sources_from_tool_manager(self, mock_ai_class, test_config):
        """Test that sources are retrieved from tool manager"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        # Manually set sources on the search tool
        rag.search_tool.last_sources = [{"title": "Test Source", "url": "http://test.com"}]

        _, sources = rag.query("Query")

        assert len(sources) == 1
        assert sources[0]["title"] == "Test Source"

    @patch("rag_system.AIGenerator")
    def test_query_resets_sources_after_retrieval(self, mock_ai_class, test_config):
        """Test that sources are reset after being retrieved"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        # Set sources
        rag.search_tool.last_sources = [{"title": "Source", "url": None}]

        rag.query("Query")

        # Sources should be reset
        assert rag.tool_manager.get_last_sources() == []

    @patch("rag_system.AIGenerator")
    def test_query_returns_empty_sources_when_none(self, mock_ai_class, test_config):
        """Test that empty sources list returned when no sources"""
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_class.return_value = mock_ai_instance

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        _, sources = rag.query("General question")

        assert sources == []


class TestRAGSystemInitialization:
    """Tests for RAGSystem initialization"""

    @patch("rag_system.AIGenerator")
    def test_initializes_all_components(self, mock_ai_class, test_config):
        """Test that all components are initialized"""
        mock_ai_class.return_value = MagicMock()

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None

    @patch("rag_system.AIGenerator")
    def test_registers_search_tools(self, mock_ai_class, test_config):
        """Test that search and outline tools are registered"""
        mock_ai_class.return_value = MagicMock()

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        tool_defs = rag.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_defs]

        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemCourseAnalytics:
    """Tests for get_course_analytics method"""

    @patch("rag_system.AIGenerator")
    def test_get_course_analytics_returns_dict(self, mock_ai_class, test_config):
        """Test that analytics returns dictionary"""
        mock_ai_class.return_value = MagicMock()

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        analytics = rag.get_course_analytics()

        assert isinstance(analytics, dict)
        assert "total_courses" in analytics
        assert "course_titles" in analytics

    @patch("rag_system.AIGenerator")
    def test_get_course_analytics_initially_empty(self, mock_ai_class, test_config):
        """Test that analytics shows zero courses initially"""
        mock_ai_class.return_value = MagicMock()

        from rag_system import RAGSystem

        rag = RAGSystem(test_config)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []
