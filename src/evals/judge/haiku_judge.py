"""Anthropic Haiku 4.5 judge.

Used by tasks that need an LLM-as-judge: sycophancy_slava, sycophancy_aisi,
spiralbench_mini. Returns a JSON object validated against a JSON schema.

The schema enforcement is done with the Anthropic SDK's `tools` API rather
than regex-parsing free-form output: the model is given a single tool whose
input matches the schema, and its `tool_use` block is the verdict.
"""
from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic

DEFAULT_MODEL = os.environ.get("HELDOUT_JUDGE_MODEL", "claude-haiku-4-5-20251001")


class HaikuJudge:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        self.client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, system: str, user: str, schema: dict) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[
                {
                    "name": "submit_verdict",
                    "description": "Submit your verdict according to the schema.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "submit_verdict"},
            messages=[{"role": "user", "content": user}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_verdict":
                return dict(block.input)
        raise RuntimeError(f"judge returned no tool_use block: {response}")
