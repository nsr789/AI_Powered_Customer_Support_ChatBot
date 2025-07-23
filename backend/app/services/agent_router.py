# ... imports unchanged ...
from langgraph.graph import END, StateGraph

# --------------------------------------------------------------------------- #
# Build LangGraph with explicit schema
# --------------------------------------------------------------------------- #
graph = StateGraph(dict)                     # ← PASS STATE SCHEMA

graph.add_node("ask_llm", ask_llm)
graph.add_node("search", run_search)
graph.add_node("fallback", run_recommender)

graph.set_entry_point("ask_llm")

graph.add_conditional_edges(
    "ask_llm",
    decide_retrieval,
    {
        "search": "search",
        "fallback": "fallback",
    },
)

graph.add_edge("search", END)
graph.add_edge("fallback", END)

router = graph.compile()
