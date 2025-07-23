"""
LLM provider utilities: wraps OpenAIChat + cheap FakeLLM for tests.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


class LLMInterface:
    """Protocol-like minimal interface."""

    def stream(self, messages: List[Dict[str, str]]):  # pragma: no cover
        raise NotImplementedError


class OpenAIProvider(LLMInterface):
    def __init__(self, temperature: float = 0.2):
        self._chat = ChatOpenAI(
            temperature=temperature,
            model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo-0125"),
        )

    def stream(self, messages: List[Dict[str, str]]):
        # Convert dict → LangChain message objects
        lc_msgs: List[BaseMessage] = []
        role_map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        for msg in messages:
            lc_msgs.append(role_map[msg["role"]](content=msg["content"]))

        for chunk in self._chat.stream(lc_msgs):
            yield chunk.content  # str pieces


class FakeLLM(LLMInterface):
    """Deterministic single-shot LLM for tests."""

    def stream(self, messages: List[Dict[str, str]]):
        prompt = messages[-1]["content"].lower()
        if "recommend" in prompt:
            yield "Here are some picks just for you!"
        else:
            yield "Sorry, I couldn't find anything."
