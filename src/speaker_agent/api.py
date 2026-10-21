from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .agent import MAX_MODEL_TURNS, AnswerProcessingError, ClaimAgent
from .data import (
    DATASET_VERSION,
    EVALUATOR_VERSION,
    FACT_VERSION,
    file_hash,
    load_answers,
    load_facts,
    load_gold,
)
from .db import Database
from .facts import FactStore
from .provider import GeminiProvider, OllamaProvider, OpenAIProvider
from .scoring import check_summary, score_development
from .validation import Claim

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
store = FactStore(load_facts(ROOT / "brand_facts.csv"))
app = FastAPI(title="Speaker claim agent")


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
    agent_version: str = Field(pattern="^v[12]$")


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_version: str = Field(pattern="^v[12]$")
    split: str = Field(pattern="^(development|check)$")


def get_agent(version: str) -> ClaimAgent:
    provider_name = os.getenv("AGENT_PROVIDER", "openai").lower()
    if provider_name not in {"openai", "gemini", "ollama"}:
        raise ValueError(
            f"Unsupported AGENT_PROVIDER={provider_name!r}; expected openai, gemini, or ollama"
        )
    timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))
    if provider_name == "openai":
        provider = OpenAIProvider(
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            timeout,
            os.getenv("OPENAI_API_KEY"),
        )
    elif provider_name == "ollama":
        provider = OllamaProvider(
            os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            timeout,
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    elif provider_name == "gemini":
        provider = GeminiProvider(
            os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            timeout,
            os.getenv("GEMINI_API_KEY"),
        )
    return ClaimAgent(
        provider,
        store,
        version,
        max_turns=MAX_MODEL_TURNS,
    )


def _run_metadata(version: str, split: str) -> dict[str, Any]:
    provider_name = os.getenv("AGENT_PROVIDER", "openai").lower()
    if provider_name == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    elif provider_name == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))
    max_turns = MAX_MODEL_TURNS
    return {
        "source": "api",
        "split": split,
        "agent_version": version,
        "dataset_version": DATASET_VERSION,
        "fact_version": FACT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "provider": provider_name,
        "model": model,
        "settings": {"timeout_seconds": timeout, "max_turns": max_turns},
        "source_revision": os.getenv("SOURCE_REVISION"),
        "input_hashes": {
            name: file_hash(ROOT / name)
            for name in ("answers.csv", "brand_facts.csv", "development_gold.csv")
        },
    }


@app.post("/classify")
def classify(request: ClassifyRequest) -> dict[str, Any]:
    if request.brand_id not in store.brands:
        raise HTTPException(404, "Unknown primary brand")
    try:
        claims, _ = get_agent(request.agent_version).classify(request.answer_id, request.brand_id, request.question, request.ai_answer)
    except AnswerProcessingError as exc:
        status = 504 if exc.kind == "turn_limit" else 502
        raise HTTPException(status, str(exc)[:200]) from exc
    return {"answer_id": request.answer_id, "claims": [claim.as_dict() for claim in claims]}


@app.post("/evaluation-runs", status_code=201)
def evaluation_runs(request: EvaluationRunRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    version, split = request.agent_version, request.split
    if not os.getenv("DATABASE_URL"):
        raise HTTPException(503, "DATABASE_URL is not configured")
    run_id = str(uuid.uuid4())
    db = Database(os.environ["DATABASE_URL"])
    metadata = _run_metadata(version, split)
    db.create_run(run_id, version, split, metadata)
    background_tasks.add_task(_process_run, run_id, version, split)
    scheduled = sum(1 for answer in load_answers(ROOT / "answers.csv") if answer.split == split)
    metrics = (
        score_development(load_gold(ROOT / "development_gold.csv"), {}, {})
        if split == "development"
        else check_summary({}, {})
    )
    return {
        "run_id": run_id,
        "status": "processing",
        "counts": {
            "scheduled_answers": scheduled,
            "completed_answers": 0,
            "successful_answers": 0,
            "error_answers": 0,
        },
        "metrics": metrics,
    }


def _process_run(run_id: str, version: str, split: str) -> None:
    """Process a fixed CSV split after the request has returned."""
    from .data import load_answers

    db = Database(os.environ["DATABASE_URL"])
    agent = get_agent(version)
    answers = [answer for answer in load_answers(ROOT / "answers.csv") if answer.split == split]
    predictions: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for answer in answers:
        try:
            claims, latency = agent.classify(answer.answer_id, answer.brand_id, answer.question, answer.ai_answer)
            predictions[answer.answer_id] = [claim.as_dict() for claim in claims]
            db.add_answer(run_id, answer.answer_id, "succeeded", round(latency * 1000), None)
            for claim in claims:
                db.add_claim(run_id, answer.answer_id, claim.as_dict())
        except AnswerProcessingError as exc:
            error = str(exc)[:500]
            errors[answer.answer_id] = error
            predictions[answer.answer_id] = []
            db.add_answer(run_id, answer.answer_id, "error", 0, error)
    db.commit()
    metrics = (
        score_development(load_gold(ROOT / "development_gold.csv"), predictions, errors)
        if split == "development"
        else check_summary(predictions, errors)
    )
    db.finish_run(run_id, "completed", metrics)


@app.get("/evaluation-runs/{run_id}")
def evaluation_run(run_id: str) -> dict[str, Any]:
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(422, "Invalid run UUID") from exc
    if not os.getenv("DATABASE_URL"):
        raise HTTPException(503, "DATABASE_URL is not configured")
    db = Database(os.environ["DATABASE_URL"])
    row = db.connection.execute("SELECT run_id, agent_version, split, status, metadata, metrics FROM evaluation_runs WHERE run_id=%s", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Run not found")
    answers = db.connection.execute("SELECT answer_id, status, latency_ms, error FROM answer_results WHERE run_id=%s ORDER BY answer_id", (run_id,)).fetchall()
    claims = db.connection.execute("SELECT answer_id, start_char, end_char, quote, label, fact_ids, reason FROM claim_results WHERE run_id=%s ORDER BY answer_id, start_char", (run_id,)).fetchall()
    return {"run": row, "answers": answers, "claims": claims}
