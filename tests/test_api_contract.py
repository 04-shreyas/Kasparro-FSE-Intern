from fastapi.testclient import TestClient

from speaker_agent import api


class EmptyAgent:
    def classify(self, answer_id, brand_id, question, ai_answer):
        return [], 0.01


class FakeDatabase:
    created = None

    def __init__(self, url):
        self.url = url

    def create_run(self, run_id, agent_version, split, metadata):
        FakeDatabase.created = (run_id, agent_version, split, metadata)


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


def test_evaluation_run_returns_counts_metrics_and_persists_metadata(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setenv("AGENT_PROVIDER", "ollama")
    monkeypatch.setattr(api, "Database", FakeDatabase)
    monkeypatch.setattr(api, "_process_run", lambda *args: None)
    client = TestClient(api.app)

    response = client.post("/evaluation-runs", json={"agent_version": "v2", "split": "check"})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "processing"
    assert body["counts"] == {
        "scheduled_answers": 10,
        "completed_answers": 0,
        "successful_answers": 0,
        "error_answers": 0,
    }
    assert body["metrics"]["joint_f1"] is None
    assert FakeDatabase.created[1:3] == ("v2", "check")
    metadata = FakeDatabase.created[3]
    assert metadata["dataset_version"] == "long-brand-answers-50-v4"
    assert metadata["provider"] == "ollama"
    assert metadata["model"] == "qwen2.5-coder:7b"
    assert set(metadata["input_hashes"]) == {
        "answers.csv",
        "brand_facts.csv",
        "development_gold.csv",
    }
