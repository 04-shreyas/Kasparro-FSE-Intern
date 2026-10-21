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


def test_evaluation_run_retrieval_returns_joined_rows_in_stable_order(monkeypatch):
    class Connection:
        def __init__(self):
            self.calls = 0

        def execute(self, query, params):
            self.calls += 1
            if self.calls == 1:
                return type("Result", (), {"fetchone": lambda self: ("run", "v1", "check", "completed", {}, {})})()
            if self.calls == 2:
                return type("Result", (), {"fetchall": lambda self: [("A01", "succeeded", 10, None)]})()
            return type("Result", (), {"fetchall": lambda self: [("A01", 0, 5, "Claim.", "unknown", [], "")]})()

    class RetrievalDatabase:
        def __init__(self, url):
            self.connection = Connection()

    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(api, "Database", RetrievalDatabase)
    response = TestClient(api.app).get("/evaluation-runs/00000000-0000-0000-0000-000000000001")
    assert response.status_code == 200
    assert response.json()["answers"] == [["A01", "succeeded", 10, None]]
    assert response.json()["claims"] == [["A01", 0, 5, "Claim.", "unknown", [], ""]]


def test_database_schema_supports_success_empty_and_error_answer_rows():
    from speaker_agent.db import Database

    url = "postgresql://postgres:postgres@127.0.0.1:5432/speaker_agent"
    run_id = "00000000-0000-0000-0000-000000000002"
    db = Database(url)
    db.connection.execute("DELETE FROM evaluation_runs WHERE run_id=%s", (run_id,))
    db.create_run(run_id, "v1", "check", {"source": "controlled-test"})
    db.add_answer(run_id, "SUCCESS", "succeeded", 10, None)
    db.add_answer(run_id, "EMPTY", "succeeded", 11, None)
    db.add_answer(run_id, "ERROR", "error", 0, "provider failed")
    db.commit()
    rows = db.connection.execute(
        "SELECT answer_id, status, error FROM answer_results WHERE run_id=%s ORDER BY answer_id",
        (run_id,),
    ).fetchall()
    assert rows == [
        ("EMPTY", "succeeded", None),
        ("ERROR", "error", "provider failed"),
        ("SUCCESS", "succeeded", None),
    ]
    db.connection.execute("DELETE FROM evaluation_runs WHERE run_id=%s", (run_id,))
    db.connection.commit()


def test_evaluation_run_processes_and_retrieves_full_background_flow(monkeypatch):
    class Result:
        def __init__(self, one=None, many=None):
            self.one = one
            self.many = many

        def fetchone(self):
            return self.one

        def fetchall(self):
            return self.many

    class Connection:
        def execute(self, query, params):
            if "SELECT run_id" in query:
                return Result(
                    one=(BackgroundDatabase.run_id, "v1", "check", "completed", {"source": "api"}, {}),
                )
            if "answer_id, status" in query:
                return Result(many=[(answer_id, "succeeded", 1, None) for answer_id in BackgroundDatabase.answers])
            return Result(many=[])

    class BackgroundDatabase(FakeDatabase):
        run_id = None
        answers = []
        claims = []

        def __init__(self, url):
            self.connection = Connection()

        def create_run(self, run_id, agent_version, split, metadata):
            BackgroundDatabase.run_id = run_id

        def add_answer(self, run_id, answer_id, status, latency_ms, error):
            BackgroundDatabase.answers.append(answer_id)

        def add_claim(self, run_id, answer_id, claim):
            BackgroundDatabase.claims.append((answer_id, claim))

        def commit(self):
            pass

        def finish_run(self, run_id, status, metrics):
            assert run_id == BackgroundDatabase.run_id
            assert status == "completed"

    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setenv("AGENT_PROVIDER", "ollama")
    monkeypatch.setattr(api, "Database", BackgroundDatabase)
    monkeypatch.setattr(api, "get_agent", lambda version: EmptyAgent())

    response = TestClient(api.app).post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert response.status_code == 201
    assert len(BackgroundDatabase.answers) == 10
    run_response = TestClient(api.app).get(f"/evaluation-runs/{BackgroundDatabase.run_id}")
    assert run_response.status_code == 200
    assert len(run_response.json()["answers"]) == 10
    assert run_response.json()["claims"] == []
