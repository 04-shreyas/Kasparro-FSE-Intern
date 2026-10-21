from __future__ import annotations

import json
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
  run_id uuid PRIMARY KEY,
  agent_version text NOT NULL,
  split text NOT NULL CHECK (split IN ('development', 'check')),
  status text NOT NULL,
  metadata jsonb NOT NULL,
  metrics jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS answer_results (
  run_id uuid NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
  answer_id text NOT NULL,
  status text NOT NULL,
  latency_ms integer,
  error text,
  PRIMARY KEY (run_id, answer_id)
);
CREATE TABLE IF NOT EXISTS claim_results (
  run_id uuid NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
  answer_id text NOT NULL,
  start_char integer NOT NULL,
  end_char integer NOT NULL,
  quote text NOT NULL,
  label text NOT NULL CHECK (label IN ('supported', 'contradicted', 'unknown')),
  fact_ids jsonb NOT NULL,
  reason text NOT NULL,
  PRIMARY KEY (run_id, answer_id, start_char, end_char),
  FOREIGN KEY (run_id, answer_id) REFERENCES answer_results(run_id, answer_id) ON DELETE CASCADE
);
"""


class Database:
    def __init__(self, url: str) -> None:
        import psycopg

        self.connection = psycopg.connect(url)
        self.connection.execute(SCHEMA)
        self.connection.commit()

    def create_run(self, run_id: str, agent_version: str, split: str, metadata: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO evaluation_runs (run_id, agent_version, split, status, metadata) VALUES (%s, %s, %s, %s, %s)",
            (run_id, agent_version, split, "processing", json.dumps(metadata)),
        )
        self.connection.commit()

    def finish_run(self, run_id: str, status: str, metrics: dict[str, Any]) -> None:
        self.connection.execute(
            "UPDATE evaluation_runs SET status=%s, metrics=%s WHERE run_id=%s",
            (status, json.dumps(metrics), run_id),
        )
        self.connection.commit()

    def add_answer(self, run_id: str, answer_id: str, status: str, latency_ms: int, error: str | None) -> None:
        self.connection.execute(
            "INSERT INTO answer_results VALUES (%s,%s,%s,%s,%s)",
            (run_id, answer_id, status, latency_ms, error),
        )

    def add_claim(self, run_id: str, answer_id: str, claim: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO claim_results VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (run_id, answer_id, claim["start_char"], claim["end_char"], claim["quote"], claim["label"], json.dumps(claim["fact_ids"]), claim["reason"]),
        )

    def commit(self) -> None:
        self.connection.commit()
