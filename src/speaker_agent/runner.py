from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .agent import AnswerProcessingError, ClaimAgent
from .agent import MAX_MODEL_TURNS
from .data import (
    DATASET_VERSION,
    EVALUATOR_VERSION,
    FACT_VERSION,
    Answer,
    file_hash,
    load_answers,
    load_facts,
    load_gold,
)
from .facts import FactStore
from .provider import GeminiProvider, OllamaProvider, OpenAIProvider
from .scoring import check_summary, score_development


def validate_answer_outcomes(scheduled_ids: list[str], outcomes: list[dict[str, Any]]) -> None:
    observed_ids = [outcome["answer_id"] for outcome in outcomes]
    if len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("Run contains duplicate answer outcomes")
    if set(observed_ids) != set(scheduled_ids):
        missing = sorted(set(scheduled_ids) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(scheduled_ids))
        raise RuntimeError(f"Run answer outcomes do not match schedule; missing={missing}, extra={extra}")


def run(project_dir: Path, version: str, split: str, output: Path) -> dict[str, Any]:
    load_dotenv(project_dir / ".env", override=True)
    answers = [x for x in load_answers(project_dir / "answers.csv") if x.split == split]
    fact_path = project_dir / "brand_facts.csv"
    store = FactStore(load_facts(fact_path))
    provider_name = os.getenv("AGENT_PROVIDER", "openai").lower()
    if provider_name not in {"openai", "gemini", "ollama"}:
        raise ValueError(
            f"Unsupported AGENT_PROVIDER={provider_name!r}; expected openai, gemini, or ollama"
        )
    timeout = float(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))
    if provider_name == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        provider = OpenAIProvider(model, timeout, os.getenv("OPENAI_API_KEY"))
    elif provider_name == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        provider = OllamaProvider(model, timeout, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    elif provider_name == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        provider = GeminiProvider(model=model, timeout=timeout, api_key=os.getenv("GEMINI_API_KEY"))
    agent = ClaimAgent(provider, store, version, max_turns=MAX_MODEL_TURNS)
    predictions: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    outcomes: list[dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": hashlib.sha256(f"{version}:{split}:{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:16],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
        "agent_version": version,
        "split": split,
        "dataset_version": DATASET_VERSION,
        "fact_version": FACT_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "provider": provider_name,
        "model": model,
        "settings": {"timeout_seconds": timeout, "max_turns": MAX_MODEL_TURNS},
        "input_hashes": {name: file_hash(project_dir / name) for name in ("answers.csv", "brand_facts.csv", "development_gold.csv")},
        "counts": {"scheduled_answers": len(answers), "completed_answers": 0, "successful_answers": 0, "error_answers": 0},
        "metrics": check_summary({}, {}) if split != "development" else score_development(load_gold(project_dir / "development_gold.csv"), {}, {}),
        "answer_outcomes": outcomes,
        "predictions": predictions,
    }

    def save_report() -> None:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save_report()
    total = len(answers)
    for index, answer in enumerate(answers, start=1):
        started = time.perf_counter()
        try:
            claims, latency = agent.classify(answer.answer_id, answer.brand_id, answer.question, answer.ai_answer)
            predictions[answer.answer_id] = [claim.as_dict() for claim in claims]
            outcomes.append({"answer_id": answer.answer_id, "status": "succeeded", "latency_seconds": latency})
        except AnswerProcessingError as exc:
            errors[answer.answer_id] = str(exc)
            predictions[answer.answer_id] = []
            outcomes.append(
                {
                    "answer_id": answer.answer_id,
                    "status": "error",
                    "error": str(exc)[:500],
                    "latency_seconds": time.perf_counter() - started,
                }
            )
        report["counts"]["completed_answers"] = index
        report["counts"]["successful_answers"] = index - len(errors)
        report["counts"]["error_answers"] = len(errors)
        report["answer_outcomes"] = outcomes
        report["predictions"] = predictions
        report["metrics"] = (
            score_development(load_gold(project_dir / "development_gold.csv"), predictions, errors)
            if split == "development"
            else check_summary(predictions, errors)
        )
        save_report()
        print(f"[{index}/{total}] {answer.answer_id}: {outcomes[-1]['status']}", flush=True)
    validate_answer_outcomes([answer.answer_id for answer in answers], outcomes)
    report["status"] = "completed"
    save_report()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--split", choices=("development", "check"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(Path.cwd(), args.version, args.split, args.output)


if __name__ == "__main__":
    main()
