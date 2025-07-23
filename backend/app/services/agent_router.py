"""
LangGraph state-machine deciding which tool to call.

Nodes:
    ask_llm → decide_retrieval → {search_products | recommender}
"""
from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

from app.core.llm import LLMInterface, OpenAIProvider, FakeLLM
from app.core.database import get_db
from app.services.indexer import search_products
from app.services.recommender import top_n

# Select LLM provider
_llm: LLMInterface = OpenAIProvider() if (os.getenv("OPENAI_API_KEY")) else FakeLLM()


def ask_llm(state: Dict) -> Dict:
    """LLM analyses user request and outputs tool name."""
    user_query = state["query"]
    # Tiny prompt; for prod you’d use a template + system role
    system_prompt = (
        "If the query is about finding or recommending a product respond "
        "with exactly 'search'. Otherwise respond with 'fallback'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    tool = "".join(_llm.stream(messages)).strip().lower()
    state["tool"] = tool
    return state


def decide_retrieval(state: Dict):
    return state["tool"]  # route key


def run_search(state: Dict):
    query = state["query"]
    db = next(get_db())
    results = search_products(db, query)
    state["results"] = [p.as_dict() for p in results]
    state["answer"] = "Here are some products I found"
    return state


def run_recommender(state: Dict):
    db = next(get_db())
    results = top_n(db)
    state["results"] = [p.as_dict() for p in results]
    state["answer"] = "Here are popular picks"
    return state


# -------- Build LangGraph ----------------------------------------------------
graph = MessageGraph()

graph.add_node("ask_llm", ask_llm)
graph.add_node("search", run_search)
graph.add_node("fallback", run_recommender)
graph.add_edge("ask_llm", decide_retrieval)

graph.set_entry_point("ask_llm")
graph.add_edge("search", END)
graph.add_edge("fallback", END)

router = graph.compile()
