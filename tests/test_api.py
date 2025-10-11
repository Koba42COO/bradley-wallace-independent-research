from fastapi.testclient import TestClient

from services.api import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_optimize_entropy():
    client = TestClient(app)
    r = client.post("/optimize", json={"mode": "entropy"})
    assert r.status_code == 200
    assert "system_metrics" in r.json()


