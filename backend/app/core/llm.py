"""
LLM provider utilities: wraps OpenAIChat + cheap FakeLLM for tests.
"""
from __future__ import annotations

import os
from typing import Dict, List

# UPDATED import path -----------------------------------------------
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

# -------------------------------------------------------------------
class LLMInterface:
    """Protocol-like minimal interface."""

    def stream(self, messages: List[Dict[str, str]]):
        raise NotImplementedError


class OpenAIProvider(LLMInterface):
    def __init__(self, temperature: float = 0.2):
        self._chat = ChatOpenAI(
            temperature=temperature,
            model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo-0125"),
        )

    def stream(self, messages: List[Dict[str, str]]):
        role_map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        lc_msgs: List[BaseMessage] = [role_map[m["role"]](content=m["content"]) for m in messages]

        for chunk in self._chat.stream(lc_msgs):
            yield chunk.content


class FakeLLM(LLMInterface):
    """Deterministic LLM for unit tests."""

    def stream(self, messages: List[Dict[str, str]]):
        prompt = messages[-1]["content"].lower()
        yield "Here are some picks" if "recommend" in prompt else "Sorry, nothing found."
