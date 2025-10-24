"""Simple agent that proxies conversation to an OpenAI chat model."""

from __future__ import annotations

import logging
import os

from collections.abc import AsyncIterable
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


class AirbnbAgent:
    """Simple conversational agent that relies on an OpenAI chat model."""

    SYSTEM_INSTRUCTION = (
        "You are a helpful assistant. Provide clear and concise answers using "
        "only the information from this conversation."
    )

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model or os.getenv(
            "OPENAI_AIRBNB_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini")
        )
        self._session_history: dict[str, list[ChatCompletionMessageParam]] = {}
        self._function_specs: list[dict[str, Any]] | None = None
        # Example: enable function calling with two mock functions.
        # self._function_specs = [
        #     {
        #         "name": "mock_property_search",
        #         "description": "Return pretend rental listings.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "city": {"type": "string"},
        #                 "guests": {"type": "integer", "minimum": 1},
        #             },
        #             "required": ["city"],
        #         },
        #     },
        #     {
        #         "name": "mock_price_estimate",
        #         "description": "Return a pretend nightly price estimate.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "city": {"type": "string"},
        #                 "season": {"type": "string"},
        #             },
        #         },
        #     },
        # ]
        logger.info("AirbnbAgent initialized with model %s", self._model)

    def _history_for_session(self, session_id: str) -> list[ChatCompletionMessageParam]:
        return self._session_history.setdefault(session_id, [])

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        """Invoke the agent for the provided query.

        Args:
            query: User prompt text.
            session_id: Identifier for maintaining conversation context.

        Returns:
            Structured payload used by the executor layer.
        """
        logger.debug(
            "AirbnbAgent.ainvoke query=%s session_id=%s", query, session_id
        )
        history = self._history_for_session(session_id)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            *history,
            {"role": "user", "content": query},
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
        )
        choice = response.choices[0].message
        content = choice.content or "I'm sorry, I was unable to generate a response."

        history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": content},
            ]
        )

        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": content,
        }

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """Asynchronous generator that yields a single response event."""
        yield await self.ainvoke(query, session_id)

    async def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Placeholder retained for future tool integrations."""

        # Example: connect the mock functions defined in ``self._function_specs``.
        # mock_tools = {
        #     "mock_property_search": lambda **kwargs: "Found 3 lovely rentals.",
        #     "mock_price_estimate": lambda **kwargs: "$150 per night.",
        # }
        # tool = mock_tools.get(name)
        # if tool is None:
        #     return f"Tool '{name}' is not supported."
        # return tool(**arguments)

        logger.debug("Tool calling is disabled. name=%s arguments=%s", name, arguments)
        return "Tool calling is currently disabled."
