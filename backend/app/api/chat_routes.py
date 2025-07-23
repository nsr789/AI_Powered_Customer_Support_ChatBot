"""
Streaming chat endpoint using Server-Sent Events (SSE).
"""
from __future__ import annotations

from typing import List

from flask import Blueprint, Response, json, request
from sse_starlette.sse import EventSourceResponse

from app.services.agent_router import router

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query: str = data.get("query", "")
    if not query:
        return {"error": "query required"}, 400

    def event_stream():
        state = {"query": query}
        for step in router.stream(state):
            if "answer" in step:
                payload = json.dumps(
                    {"answer": step["answer"], "results": step.get("results", [])}
                )
                yield f"data: {payload}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")
