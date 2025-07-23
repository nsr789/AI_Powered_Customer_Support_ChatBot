from app.services.agent_router import router

def test_fake_llm_recommender():
    state = router.invoke({"query": "Give me some recommendations"})
    assert state["tool"] == "fallback"
    assert "results" in state and len(state["results"]) == 5
