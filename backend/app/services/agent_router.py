"""
LangGraph state-machine deciding which tool to call.
"""
from __future__ import annotations

import os
from typing import Dict, List

from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

from app.core.llm import LLMInterface, OpenAIProvider, FakeLLM
from app.core.database import get_db
from app.services.indexer import search_products
from app.services.recommender import top_n

# --------------------------------------------------------------------------- #
# Select LLM provider
# --------------------------------------------------------------------------- #
_llm: LLMInterface = OpenAIProvider() if os.getenv("OPENAI_API_KEY") else FakeLLM()

# --------------------------------------------------------------------------- #
# Node functions
# --------------------------------------------------------------------------- #
def ask_llm(state: Dict) -> Dict:
    """LLM analyses user request and writes `tool` into state."""
    user_query = state["query"]
    system_prompt = (
        "If the query is about finding or recommending a product respond "
        "with exactly 'search'. Otherwise respond with 'fallback'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    tool = "".join(_llm.stream(messages)).strip().lower()
    state["tool"] = "search" if tool == "search" else "fallback"
    return state


def decide_retrieval(state: Dict) -> str:
    """Return next node key based on `tool`."""
    return state["tool"]


def run_search(state: Dict) -> Dict:
    query = state["query"]
    db = next(get_db())
    results = search_products(db, query)
    state.update(
        {
            "results": [p.as_dict() for p in results],
            "answer": "Here are some products I found.",
        }
    )
    return state


def run_recommender(state: Dict) -> Dict:
    db = next(get_db())
    results = top_n(db)
    state.update(
        {
            "results": [p.as_dict() for p in results],
            "answer": "Here are popular picks for you.",
        }
    )
    return state


# --------------------------------------------------------------------------- #
# Build LangGraph
# --------------------------------------------------------------------------- #
graph = MessageGraph()

graph.add_node("ask_llm", ask_llm)
graph.add_node("search", run_search)
graph.add_node("fallback", run_recommender)

graph.set_entry_point("ask_llm")

# Correct conditional routing
graph.add_conditional_edges(
    source="ask_llm",
    condition=decide_retrieval,
    path_map={
        "search": "search",
        "fallback": "fallback",
    },
)

graph.add_edge("search", END)
graph.add_edge("fallback", END)

router = graph.compile()
