import anthropic
from typing import List, Optional, Dict, Any
from config import config

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search and outline tools.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Multi-Step Reasoning:
- For complex queries requiring multiple pieces of information, you may use tools sequentially
- First tool call: gather initial information
- Based on results, you may make ONE additional tool call if needed
- Maximum 2 tool rounds per query - plan your searches efficiently

Outline Tool Usage:
- Use the outline tool for questions about course structure, lesson lists, or "what lessons are in this course"
- Returns course title, course link, and complete lesson list with numbers and titles
- Present the outline in a clear, organized format

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Course outline/structure questions**: Use the outline tool, then present the full course title, course link, and all lesson numbers with titles
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle sequential tool execution with up to MAX_TOOL_ROUNDS rounds.

        Termination conditions:
        1. MAX_TOOL_ROUNDS reached
        2. Response has no tool_use blocks (stop_reason != "tool_use")
        3. Tool execution fails with critical error

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters (includes tools if available)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0

        while round_count < config.MAX_TOOL_ROUNDS:
            round_count += 1

            # Append assistant's tool_use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []
            execution_error = None

            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {e}",
                            "is_error": True
                        })
                        execution_error = e

            # Append tool results
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Determine if more tool rounds are allowed
            can_use_more_tools = (round_count < config.MAX_TOOL_ROUNDS) and (execution_error is None)

            # Prepare next API call
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }

            # Include tools only if more rounds are allowed
            if can_use_more_tools and base_params.get("tools"):
                api_params["tools"] = base_params["tools"]
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            current_response = self.client.messages.create(**api_params)

            # Check termination: no more tool use requested
            if current_response.stop_reason != "tool_use":
                break

        return self._extract_text_from_response(current_response)

    def _extract_text_from_response(self, response) -> str:
        """Extract text from response that may have multiple block types."""
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        return ""