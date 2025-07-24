"""
LangGraph state-machine that decides: product search, recommendations, or
customer-support answer.
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
    """Classify user request."""
    user_query = state["query"]
    system_prompt = (
        "Classify the user request strictly as one word: "
        "'search', 'recommend', or 'support'.\n"
        "- 'search': user looking for a product or keyword\n"
        "- 'recommend': user asks for suggestions / what to buy\n"
        "- 'support': user asks about order, shipping, returns, warranty, payment"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    decision = "".join(_llm.stream(messages)).strip().lower()
    state["tool"] = decision if decision in {"search", "recommend", "support"} else "support"
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


def run_recommend(state: Dict) -> Dict:
    db = next(get_db())
    results = top_n(db, limit=5)
    state.update(
        {
            "answer": "Here are popular picks for you.",
            "results": [p.as_dict() for p in results],
        }
    )
    return state


def run_support(state: Dict) -> Dict:
    state.update(support_answer(state["query"]))
    return state


# ───────────────────────── Build graph ────────────────────────────────────────
graph = StateGraph(dict)

graph.add_node("ask_llm", ask_llm)
graph.add_node("search", run_search)
graph.add_node("recommend", run_recommend)
graph.add_node("support", run_support)

graph.set_entry_point("ask_llm")

graph.add_conditional_edges(
    "ask_llm",
    decide_next,
    {"search": "search", "recommend": "recommend", "support": "support"},
)

graph.add_edge("search", END)
graph.add_edge("recommend", END)
graph.add_edge("support", END)

router = graph.compile()
