"""Tests for AIGenerator with mocked Anthropic API"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from ai_generator import AIGenerator


class TestAIGeneratorDirectResponse:
    """Tests for AIGenerator.generate_response() without tool use"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_returns_text(self, mock_anthropic):
        """Test that generate_response returns text from Claude"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Direct answer text")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(query="What is 2+2?")

        assert result == "Direct answer text"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_calls_api_once_without_tools(self, mock_anthropic):
        """Test that API is called once when no tools are used"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Simple question")

        assert mock_client.messages.create.call_count == 1

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_includes_system_prompt(self, mock_anthropic):
        """Test that system prompt is included in API call"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Question")

        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert "AI assistant" in call_args.kwargs["system"]


class TestAIGeneratorWithConversationHistory:
    """Tests for conversation history handling"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_history_included_in_system(self, mock_anthropic):
        """Test that conversation history is added to system prompt"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        history = "User: Previous question\nAssistant: Previous answer"

        generator.generate_response(query="Follow-up", conversation_history=history)

        call_args = mock_client.messages.create.call_args
        assert "Previous conversation" in call_args.kwargs["system"]
        assert "Previous question" in call_args.kwargs["system"]

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_history_when_none_provided(self, mock_anthropic):
        """Test system prompt without history section when not provided"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Question", conversation_history=None)

        call_args = mock_client.messages.create.call_args
        assert "Previous conversation" not in call_args.kwargs["system"]


class TestAIGeneratorWithToolUse:
    """Tests for tool use handling"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_tools_added_to_api_call(self, mock_anthropic):
        """Test that tools are properly added to API call"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        tools = [{"name": "search_course_content", "input_schema": {}}]

        generator.generate_response(query="Test", tools=tools)

        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic):
        """Test complete tool execution flow with two API calls"""
        mock_client = MagicMock()

        # First response: tool use
        tool_content = MagicMock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.id = "tool_123"
        tool_content.input = {"query": "test query"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_content]

        # Second response: final answer
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [MagicMock(text="Final synthesized answer")]

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic.return_value = mock_client

        # Setup mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result: found content"

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Search for something",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        assert result == "Final synthesized answer"
        assert mock_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_result_passed_to_second_call(self, mock_anthropic):
        """Test that tool results are included in follow-up API call"""
        mock_client = MagicMock()

        # First response: tool use
        tool_content = MagicMock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.id = "tool_456"
        tool_content.input = {"query": "MCP"}

        first_response = MagicMock()
        first_response.stop_reason = "tool_use"
        first_response.content = [tool_content]

        # Second response: final answer
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [MagicMock(text="Answer")]

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results: MCP content"

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Check second API call has tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs["messages"]

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_456"
        assert "Search results: MCP content" in messages[2]["content"][0]["content"]


class TestAIGeneratorHandleToolExecution:
    """Tests for _handle_tool_execution method"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_returns_final_text(self, mock_anthropic):
        """Test that _handle_tool_execution returns final response text"""
        mock_client = MagicMock()
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Final answer")]
        mock_client.messages.create.return_value = final_response
        mock_anthropic.return_value = mock_client

        # Create tool use response
        tool_content = MagicMock()
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.id = "tool_789"
        tool_content.input = {"query": "test"}

        initial_response = MagicMock()
        initial_response.content = [tool_content]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool output"

        generator = AIGenerator(api_key="test-key", model="test-model")
        base_params = {
            "messages": [{"role": "user", "content": "Query"}],
            "system": "System prompt"
        }

        result = generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        assert result == "Final answer"

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_multiple_tool_calls(self, mock_anthropic):
        """Test handling multiple tool calls in one response"""
        mock_client = MagicMock()
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Combined answer")]
        mock_client.messages.create.return_value = final_response
        mock_anthropic.return_value = mock_client

        # Create response with multiple tool calls
        tool_content1 = MagicMock()
        tool_content1.type = "tool_use"
        tool_content1.name = "search_course_content"
        tool_content1.id = "tool_1"
        tool_content1.input = {"query": "first query"}

        tool_content2 = MagicMock()
        tool_content2.type = "tool_use"
        tool_content2.name = "get_course_outline"
        tool_content2.id = "tool_2"
        tool_content2.input = {"course_name": "Test Course"}

        initial_response = MagicMock()
        initial_response.content = [tool_content1, tool_content2]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="test-model")
        base_params = {
            "messages": [{"role": "user", "content": "Query"}],
            "system": "System prompt"
        }

        result = generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        assert result == "Combined answer"
        assert mock_tool_manager.execute_tool.call_count == 2


class TestAIGeneratorAPIParameters:
    """Tests for API call parameters"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_temperature_is_zero(self, mock_anthropic):
        """Test that temperature is set to 0 for consistent responses"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Question")

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["temperature"] == 0

    @patch('ai_generator.anthropic.Anthropic')
    def test_max_tokens_is_set(self, mock_anthropic):
        """Test that max_tokens is set"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(query="Question")

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_model_is_passed(self, mock_anthropic):
        """Test that model name is passed to API"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-test-model")
        generator.generate_response(query="Question")

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-test-model"


class TestAIGeneratorSequentialToolCalls:
    """Tests for sequential tool calling (up to 2 rounds)"""

    def _create_tool_use_response(self, tool_name, tool_input, tool_id):
        """Helper to create a mock tool use response"""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.id = tool_id
        tool_block.input = tool_input

        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [tool_block]
        return response

    def _create_text_response(self, text):
        """Helper to create a mock text response"""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text

        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [text_block]
        return response

    @patch('ai_generator.anthropic.Anthropic')
    def test_two_sequential_tool_calls(self, mock_anthropic):
        """Test complete flow with two sequential tool calls (3 API calls)"""
        mock_client = MagicMock()

        # Response 1: First tool call
        tool_call_1 = self._create_tool_use_response(
            "get_course_outline", {"course_name": "Course A"}, "tool_1"
        )

        # Response 2: Second tool call (after seeing first results)
        tool_call_2 = self._create_tool_use_response(
            "search_course_content", {"query": "lesson 4 topic"}, "tool_2"
        )

        # Response 3: Final answer
        final_response = self._create_text_response("Complete answer about the topic")

        mock_client.messages.create.side_effect = [tool_call_1, tool_call_2, final_response]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Outline: Lesson 4 is about MCP", "MCP content found"]

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Find courses about the same topic as lesson 4",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        assert result == "Complete answer about the topic"
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_single_tool_round_terminates_early(self, mock_anthropic):
        """Test that loop terminates when Claude answers after 1 tool call"""
        mock_client = MagicMock()

        # Response 1: Tool call
        tool_call = self._create_tool_use_response(
            "search_course_content", {"query": "MCP"}, "tool_1"
        )

        # Response 2: Final answer (no more tool use)
        final_response = self._create_text_response("Answer based on search")

        mock_client.messages.create.side_effect = [tool_call, final_response]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results"

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        assert result == "Answer based on search"
        assert mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    @patch('ai_generator.anthropic.Anthropic')
    def test_max_rounds_enforced(self, mock_anthropic):
        """Test that max rounds limit is enforced (tools removed after limit)"""
        mock_client = MagicMock()

        # All responses want more tool use
        tool_call_1 = self._create_tool_use_response("search_course_content", {"query": "q1"}, "t1")
        tool_call_2 = self._create_tool_use_response("search_course_content", {"query": "q2"}, "t2")

        # Third response after max rounds (forced without tools)
        final_response = self._create_text_response("Final answer after max rounds")

        mock_client.messages.create.side_effect = [tool_call_1, tool_call_2, final_response]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Complex query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        assert result == "Final answer after max rounds"
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_tools_available_in_second_round(self, mock_anthropic):
        """Test that tools are included in second API call (before max rounds)"""
        mock_client = MagicMock()

        tool_call_1 = self._create_tool_use_response("search_course_content", {"query": "q1"}, "t1")
        final_response = self._create_text_response("Answer")

        mock_client.messages.create.side_effect = [tool_call_1, final_response]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator = AIGenerator(api_key="test-key", model="test-model")
        tools = [{"name": "search_course_content"}]
        generator.generate_response(query="Query", tools=tools, tool_manager=mock_tool_manager)

        # Check second API call includes tools (round 1 < max 2)
        second_call = mock_client.messages.create.call_args_list[1]
        assert "tools" in second_call.kwargs
        assert second_call.kwargs["tools"] == tools

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_handled(self, mock_anthropic):
        """Test that tool execution errors are captured and loop stops further tool rounds"""
        mock_client = MagicMock()

        tool_call = self._create_tool_use_response("search_course_content", {"query": "q1"}, "t1")
        final_response = self._create_text_response("Response after error")

        mock_client.messages.create.side_effect = [tool_call, final_response]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

        generator = AIGenerator(api_key="test-key", model="test-model")
        result = generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        assert result == "Response after error"
        # Second call should NOT have tools (error stops further rounds)
        second_call = mock_client.messages.create.call_args_list[1]
        assert "tools" not in second_call.kwargs

    @patch('ai_generator.anthropic.Anthropic')
    def test_messages_accumulate_correctly(self, mock_anthropic):
        """Test that messages accumulate correctly after 2 tool rounds"""
        mock_client = MagicMock()

        tool_call_1 = self._create_tool_use_response("search_course_content", {"query": "q1"}, "t1")
        tool_call_2 = self._create_tool_use_response("search_course_content", {"query": "q2"}, "t2")
        final_response = self._create_text_response("Final")

        mock_client.messages.create.side_effect = [tool_call_1, tool_call_2, final_response]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Check final API call has accumulated messages
        final_call = mock_client.messages.create.call_args_list[2]
        messages = final_call.kwargs["messages"]

        # Should have: user, assistant(tool1), user(result1), assistant(tool2), user(result2)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_tools_in_final_call_after_max_rounds(self, mock_anthropic):
        """Test that tools are removed in the call after max rounds reached"""
        mock_client = MagicMock()

        tool_call_1 = self._create_tool_use_response("search_course_content", {"query": "q1"}, "t1")
        tool_call_2 = self._create_tool_use_response("search_course_content", {"query": "q2"}, "t2")
        final_response = self._create_text_response("Final")

        mock_client.messages.create.side_effect = [tool_call_1, tool_call_2, final_response]
        mock_anthropic.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="test-model")
        generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Third call (after max rounds) should NOT have tools
        third_call = mock_client.messages.create.call_args_list[2]
        assert "tools" not in third_call.kwargs
