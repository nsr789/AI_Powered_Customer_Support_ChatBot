from app.services.support_loader import main as ingest
from app.services.agent_router import router

def test_support_answer():
    ingest()  # idempotent
    state = router.invoke({"query": "What is the return policy?"})
    assert state["tool"] == "support"
    assert "return" in state["answer"].lower()
    assert state["sources"]  # at least one doc metadata
