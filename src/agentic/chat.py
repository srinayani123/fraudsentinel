"""
Conversational chat agent for the dashboard.

Wraps the same tool palette as the orchestrator but in a free-form chat loop.
Maintains conversation history. Used by the Investigation page chat sidebar.
"""

from __future__ import annotations

import json
from typing import Generator

import anthropic

from src.agentic.prompts import CHAT_SYSTEM_PROMPT
from src.agentic.tools import TOOL_SCHEMAS, execute_tool
from src.utils.config import DEFAULT_ANTHROPIC_MODEL


class ChatAgent:
    """Stateful chat agent that can call tools across turns."""

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL):
        if not api_key:
            raise ValueError("Anthropic API key required")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.history: list[dict] = []
        self.context_payload: dict = {}

    def set_context(self, transaction: dict, scores: dict, report: str | None = None):
        """Inject the current transaction context. Called when user opens a transaction."""
        self.context_payload = {
            "transaction": transaction,
            "scores": scores,
            "report": report,
        }
        # Reset history when context changes
        self.history = []

    def _build_system_prompt(self) -> str:
        ctx_str = ""
        if self.context_payload:
            ctx_str = (
                "\n\nCURRENT TRANSACTION UNDER REVIEW:\n"
                + json.dumps(self.context_payload, indent=2, default=str)
            )
        return CHAT_SYSTEM_PROMPT + ctx_str

    def chat_stream(self, user_message: str) -> Generator[dict, None, None]:
        """Stream events: text deltas + tool-use indicators."""
        self.history.append({"role": "user", "content": user_message})

        for _ in range(5):  # tool-use rounds
            assistant_text = ""
            tool_uses = []  # accumulate tool-use blocks for this turn
            assistant_content = []  # what we'll save in history

            with self.client.messages.stream(
                model=self.model,
                max_tokens=2000,
                system=self._build_system_prompt(),
                tools=TOOL_SCHEMAS,
                messages=self.history,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            tool_uses.append(
                                {
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                    "input_str": "",
                                }
                            )
                            yield {
                                "type": "tool_call_start",
                                "tool": event.content_block.name,
                            }
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            assistant_text += event.delta.text
                            yield {"type": "text_delta", "text": event.delta.text}
                        elif event.delta.type == "input_json_delta":
                            if tool_uses:
                                tool_uses[-1]["input_str"] += event.delta.partial_json

                final_message = stream.get_final_message()
                stop_reason = final_message.stop_reason
                assistant_content = final_message.content

            self.history.append({"role": "assistant", "content": assistant_content})

            if stop_reason == "tool_use":
                # Execute each tool, build tool_result content
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input)
                        yield {
                            "type": "tool_call_done",
                            "tool": block.name,
                            "input": block.input,
                            "output": result,
                        }
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result, default=str),
                            }
                        )
                self.history.append({"role": "user", "content": tool_results})
                continue

            # End of turn
            yield {"type": "done", "text": assistant_text}
            return

        yield {"type": "done", "text": "Chat hit max tool rounds."}
