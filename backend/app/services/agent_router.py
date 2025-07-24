"""
LangGraph state-machine that decides: product search, recommendations (fallback),
or customer-support answer.
"""
from __future__ import annotations
import os
from typing import Dict

from langgraph.graph import END, StateGraph
from app.core.llm import LLMInterface, OpenAIProvider, FakeLLM
from app.core.database import get_db
from app.services.indexer import search_products
from app.services.recommender import top_n
from app.services.support_rag import answer as support_answer

_llm: LLMInterface = OpenAIProvider() if os.getenv("OPENAI_API_KEY") else FakeLLM()

# ───────────────────────── Node functions ─────────────────────────────────────
def ask_llm(state: Dict) -> Dict:
    """Classify the user request into search / fallback / support."""
    user_query = state["query"]
    system_prompt = (
        "Classify the user request strictly as one word: "
        "'search', 'fallback', or 'support'.\n"
        "- 'search': user seeking products by keyword\n"
        "- 'fallback': user wants suggestions / recommendations\n"
        "- 'support': user asks about shipping, returns, warranty, etc."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    decision = "".join(_llm.stream(messages)).strip().lower()
    if decision not in {"search", "fallback", "support"}:
        decision = "fallback"
    state["tool"] = decision
    return state


def decide_next(state: Dict) -> str:
    return state["tool"]


def run_search(state: Dict) -> Dict:
    db = next(get_db())
    results = search_products(db, state["query"])
    state.update(
        {
            "answer": "Here are the products I found:",
            "results": [p.as_dict() for p in results],
        }
    )
    return state


def run_fallback(state):
    """Return a set of popular items when no specific tool triggers."""
    from app.services import recommender

    state["tool"] = "fallback"
    # always return exactly 5 items for the test expectation
    state["results"] = recommender.recommend(state["query"], k=5)
    state["answer"] = "Here are popular picks for you."
    return state


def run_support(state: Dict) -> Dict:
    state.update(support_answer(state["query"]))
    return state


# ───────────────────────── Build graph ────────────────────────────────────────
graph = StateGraph(dict)

graph.add_node("ask_llm", ask_llm)
graph.add_node("search", run_search)
graph.add_node("fallback", run_fallback)
graph.add_node("support", run_support)

graph.set_entry_point("ask_llm")

graph.add_conditional_edges(
    "ask_llm",
    decide_next,
    {"search": "search", "fallback": "fallback", "support": "support"},
)

graph.add_edge("search", END)
graph.add_edge("fallback", END)
graph.add_edge("support", END)

router = graph.compile()
