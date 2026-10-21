from types import SimpleNamespace

from fastapi.testclient import TestClient

from speaker_agent import api


class EmptyAgent:
    def classify(self, answer_id, brand_id, question, ai_answer):
        return [], 0.01


def test_classify_rejects_unknown_brand(monkeypatch):
    client = TestClient(api.app)
    response = client.post(
        "/classify",
        json={
            "answer_id": "A01",
            "brand_id": "not-a-brand",
            "question": "q",
            "ai_answer": "a",
            "agent_version": "v1",
        },
    )
    assert response.status_code == 404


def test_classify_returns_valid_empty_result(monkeypatch):
    monkeypatch.setattr(api, "get_agent", lambda version: EmptyAgent())
    client = TestClient(api.app)
    response = client.post(
        "/classify",
        json={
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "q",
            "ai_answer": "a",
            "agent_version": "v1",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"answer_id": "A01", "claims": []}


def test_evaluation_run_reports_missing_database_configuration(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    client = TestClient(api.app)
    response = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert response.status_code == 503
