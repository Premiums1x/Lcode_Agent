"""ReAct Agent implementation.

ReAct = Reasoning + Acting paradigm.
The agent thinks, decides to use a tool, observes the result,
and repeats until it has a final answer.
"""

import json
from typing import Any

from lcode.agents.base import BaseAgent
from lcode.llm.base import LLMResponse, Message
from lcode.tools.registry import ToolRegistry


class ReActAgent(BaseAgent):
    """An agent using the ReAct (Reasoning + Acting) paradigm.

    This is the Level 2 implementation.
    It can reason about what tool to use, execute it,
    and incorporate the result back into the conversation.
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        tool_registry: ToolRegistry,
        system_prompt: str = "You are a helpful assistant that can use tools.",
        max_iterations: int = 10,
    ) -> None:
        super().__init__(name, llm, system_prompt)
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations

    async def run(self, user_input: str, **kwargs: Any) -> LLMResponse:
        """Run the ReAct loop.

        The agent will:
        1. Think about what to do
        2. Call a tool (if needed)
        3. Observe the result
        4. Repeat until answer or max iterations
        """
        # Add user input to history
        self.add_to_history("user", user_input)

        # Build messages with ReAct instructions
        messages = self._build_react_messages()

        for _ in range(self.max_iterations):
            # Call LLM with tools
            tools = self.tool_registry.to_openai_schemas()
            response = await self._call_llm(messages, tools=tools, **kwargs)

            # Check if tool calls were made
            if response.tool_calls:
                # Add assistant message with tool calls
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)
                self.message_history.append(assistant_msg)

                # Execute each tool call
                for tc in response.tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]
                    tool_id = tc["id"]

                    # Execute tool
                    try:
                        result = await self.tool_registry.execute(tool_name, tool_args)
                        observation = str(result)
                    except Exception as e:
                        observation = f"Error: {e}"

                    # Add tool result message
                    tool_msg = Message(
                        role="tool",
                        content=observation,
                        tool_call_id=tool_id,
                    )
                    messages.append(tool_msg)
                    self.message_history.append(tool_msg)
            else:
                # No tool call - final answer
                self.add_to_history("assistant", response.content)
                return response

        # Max iterations reached
        return LLMResponse(
            content="I reached the maximum number of iterations without finding a complete answer. Please try rephrasing your question.",
            model="unknown",
        )

    def _build_react_messages(self) -> list[Message]:
        """Build messages with ReAct system prompt."""
        tools_desc = []
        for tool in self.tool_registry.list_tools():
            tools_desc.append(f"- {tool.name}: {tool.description}")

        tools_text = "\n".join(tools_desc)
        system = (
            f"{self.system_prompt}\n\n"
            "You have access to the following tools:\n"
            f"{tools_text}\n\n"
            "When you need to use a tool, respond with a tool call. "
            "After receiving the tool result, analyze it and either "
            "make another tool call or provide your final answer."
        )

        messages = [Message(role="system", content=system)]
        # Include full history (which already has the user input)
        messages.extend(self.message_history)
        return messages
