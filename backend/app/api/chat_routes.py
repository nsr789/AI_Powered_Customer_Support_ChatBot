"""
Streaming chat endpoint using Server-Sent Events (SSE).
"""
from __future__ import annotations

import json
from flask import Blueprint, Response, request

from app.services.agent_router import router

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query: str = data.get("query", "").strip()
    if not query:
        return {"error": "query required"}, 400

    # ---------- single-tick execution ----------
    final_state = router.invoke({"query": query})

    def event_stream():
        payload = json.dumps(
            {
                "answer": final_state["answer"],
                "results": final_state.get("results", []),
            }
        )
        yield f"data: {payload}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")
