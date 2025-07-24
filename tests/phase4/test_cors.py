from app.main import create_app

def test_cors_header():
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.headers["Access-Control-Allow-Origin"] == "*"
