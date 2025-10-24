"""Weather agent that proxies requests to an OpenAI chat model."""

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


class WeatherAgent:
    """Interact with an OpenAI chat model to answer weather questions."""

    SYSTEM_INSTRUCTION = (
        "You are a helpful weather assistant. Respond clearly and concisely using only "
        "the information you have been provided in the conversation."
    )

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model or os.getenv(
            "OPENAI_WEATHER_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini")
        )
        self._session_history: dict[str, list[ChatCompletionMessageParam]] = {}
        self._function_specs: list[dict[str, Any]] | None = None
        # Example: enable function calling with two mock functions.
        # self._function_specs = [
        #     {
        #         "name": "mock_weather_lookup",
        #         "description": "Return pretend forecast data for a city.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "city": {"type": "string"},
        #                 "state": {"type": "string"},
        #             },
        #             "required": ["city", "state"],
        #         },
        #     },
        #     {
        #         "name": "mock_weather_alerts",
        #         "description": "Return pretend weather alerts for a region.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "region": {"type": "string"},
        #             },
        #             "required": ["region"],
        #         },
        #     },
        # ]
        logger.info("WeatherAgent initialized with model %s", self._model)

    def _history_for_session(self, session_id: str) -> list[ChatCompletionMessageParam]:
        return self._session_history.setdefault(session_id, [])

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        logger.debug("WeatherAgent.ainvoke query=%s session_id=%s", query, session_id)
        history = self._history_for_session(session_id)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            *history,
            {"role": "user", "content": query},
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
        )
        message = response.choices[0].message
        content = message.content or "I was unable to generate a response."

        assistant_message = {"role": "assistant", "content": content}
        messages.append(assistant_message)
        history.extend([
            {"role": "user", "content": query},
            assistant_message,
        ])

        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": content,
        }

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        yield await self.ainvoke(query, session_id)

    async def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Placeholder retained for future tool integrations."""

        # Example: connect the mock functions defined in ``self._function_specs``.
        # mock_tools = {
        #     "mock_weather_lookup": lambda **kwargs: "Sunny and mild.",
        #     "mock_weather_alerts": lambda **kwargs: "No alerts at this time.",
        # }
        # tool = mock_tools.get(name)
        # if tool is None:
        #     return f"Tool '{name}' is not supported."
        # return tool(**arguments)

        logger.debug("Tool calling is disabled. name=%s arguments=%s", name, arguments)
        return "Tool calling is currently disabled."
