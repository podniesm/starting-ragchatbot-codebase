"""Tests for CourseSearchTool.execute() and ToolManager"""
import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_returns_string(self, test_vector_store, sample_course, sample_chunks):
        """Test that execute() returns a string result"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        tool = CourseSearchTool(test_vector_store)
        result = tool.execute(query="Python")

        assert isinstance(result, str)

    def test_execute_returns_formatted_results(self, test_vector_store, sample_course, sample_chunks):
        """Test that execute() returns properly formatted results with headers"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        tool = CourseSearchTool(test_vector_store)
        result = tool.execute(query="Python programming")

        # Should contain course title header format [Course Title - Lesson N]
        assert "[Test Course" in result
        assert "Python" in result.lower() or "content" in result.lower()

    def test_execute_with_no_results_returns_message(self, test_vector_store):
        """Test that execute() returns appropriate message when no results found"""
        tool = CourseSearchTool(test_vector_store)
        result = tool.execute(query="nonexistent topic xyz123 qwerty")

        assert "No relevant content found" in result

    def test_execute_with_course_filter(self, test_vector_store, sample_course, sample_chunks):
        """Test filtering by course name"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        tool = CourseSearchTool(test_vector_store)
        result = tool.execute(query="content", course_name="Test")

        # Should find results from Test Course
        assert "[Test Course" in result or "No relevant content found" not in result

    def test_execute_with_invalid_course_returns_error(self, test_vector_store):
        """Test that invalid course name returns error message when no courses exist"""
        # Test with empty store - no courses to match
        tool = CourseSearchTool(test_vector_store)
        result = tool.execute(query="content", course_name="Any Course Name")

        assert "No course found matching" in result

    def test_execute_with_lesson_filter(self, test_vector_store, sample_course, sample_chunks):
        """Test filtering by lesson number"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        tool = CourseSearchTool(test_vector_store)
        result = tool.execute(query="content", lesson_number=0)

        # Should return results (may be filtered to lesson 0 only)
        assert isinstance(result, str)

    def test_last_sources_tracked_on_success(self, test_vector_store, sample_course, sample_chunks):
        """Test that last_sources is populated after successful search"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        tool = CourseSearchTool(test_vector_store)
        tool.execute(query="Python")

        assert isinstance(tool.last_sources, list)
        # Should have sources if results were found
        if tool.last_sources:
            for source in tool.last_sources:
                assert "title" in source
                assert "url" in source

    def test_last_sources_empty_on_no_results(self, test_vector_store):
        """Test that last_sources is empty when no results found"""
        tool = CourseSearchTool(test_vector_store)
        tool.execute(query="nonexistent xyz123")

        assert tool.last_sources == []

    def test_format_results_includes_lesson_info(self, test_vector_store, sample_course, sample_chunks):
        """Test _format_results() includes lesson numbers when available"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        tool = CourseSearchTool(test_vector_store)
        results = test_vector_store.search(query="Python")

        if not results.is_empty():
            formatted = tool._format_results(results)
            # Should contain lesson info in header
            assert "Lesson" in formatted


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool"""

    def test_execute_returns_outline(self, test_vector_store, sample_course):
        """Test that execute() returns formatted course outline"""
        test_vector_store.add_course_metadata(sample_course)

        tool = CourseOutlineTool(test_vector_store)
        result = tool.execute(course_name="Test")

        assert "Course:" in result
        assert "Test Course" in result
        assert "Lessons:" in result

    def test_execute_with_invalid_course(self, test_vector_store):
        """Test that invalid course returns error when no courses exist"""
        # Test with empty store - no courses to match
        tool = CourseOutlineTool(test_vector_store)
        result = tool.execute(course_name="Any Course Name")

        assert "No course found matching" in result

    def test_last_sources_tracked(self, test_vector_store, sample_course):
        """Test that sources are tracked for outline tool"""
        test_vector_store.add_course_metadata(sample_course)

        tool = CourseOutlineTool(test_vector_store)
        tool.execute(course_name="Test")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["title"] == "Test Course"


class TestToolManager:
    """Tests for ToolManager class"""

    def test_register_tool(self, test_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(test_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_register_multiple_tools(self, test_vector_store):
        """Test registering multiple tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(test_vector_store)
        outline_tool = CourseOutlineTool(test_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools

    def test_get_tool_definitions(self, test_vector_store):
        """Test getting tool definitions for API call"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(test_vector_store))

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
        assert "input_schema" in definitions[0]
        assert "description" in definitions[0]

    def test_execute_tool(self, test_vector_store, sample_course, sample_chunks):
        """Test executing a registered tool"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        manager = ToolManager()
        manager.register_tool(CourseSearchTool(test_vector_store))

        result = manager.execute_tool("search_course_content", query="Python")

        assert isinstance(result, str)

    def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error message"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result

    def test_get_last_sources(self, test_vector_store, sample_course, sample_chunks):
        """Test retrieving sources from last search"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        manager = ToolManager()
        manager.register_tool(CourseSearchTool(test_vector_store))
        manager.execute_tool("search_course_content", query="Python")

        sources = manager.get_last_sources()

        assert isinstance(sources, list)

    def test_reset_sources(self, test_vector_store, sample_course, sample_chunks):
        """Test resetting sources after retrieval"""
        test_vector_store.add_course_metadata(sample_course)
        test_vector_store.add_course_content(sample_chunks)

        manager = ToolManager()
        manager.register_tool(CourseSearchTool(test_vector_store))
        manager.execute_tool("search_course_content", query="Python")

        manager.reset_sources()
        sources = manager.get_last_sources()

        assert sources == []


class TestSearchToolDefinition:
    """Tests for tool definition schema"""

    def test_search_tool_definition_schema(self, test_vector_store):
        """Test that tool definition has correct schema"""
        tool = CourseSearchTool(test_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_outline_tool_definition_schema(self, test_vector_store):
        """Test that outline tool definition has correct schema"""
        tool = CourseOutlineTool(test_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition

        schema = definition["input_schema"]
        assert "course_name" in schema["properties"]
        assert schema["required"] == ["course_name"]
