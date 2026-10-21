import csv
import json

import pytest

from speaker_agent.agent import MAX_MODEL_TURNS, AnswerProcessingError, ClaimAgent
from speaker_agent.data import load_answers
from speaker_agent.facts import FactStore
from speaker_agent.provider import ProviderError
from speaker_agent.validation import OutputValidationError, validate_output


class Response:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content, "tool_calls": []})()})()]


class FailingProvider:
    def complete(self, messages, tools):
        raise ProviderError("request timed out")


def test_csv_parser_preserves_quoted_commas_and_newlines(tmp_path):
    path = tmp_path / "answers.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["answer_id", "split", "brand_id", "question", "ai_answer"])
        writer.writeheader()
        writer.writerow({
            "answer_id": "A1",
            "split": "development",
            "brand_id": "cleartrail",
            "question": "Which model?",
            "ai_answer": "It costs INR 10,000, including delivery.\nIt is supported.",
        })
    answer = load_answers(path)[0]
    assert answer.ai_answer == "It costs INR 10,000, including delivery.\nIt is supported."


def test_validation_rejects_bad_offsets_and_quote_mismatch():
    with pytest.raises(OutputValidationError):
        validate_output(
            "A1",
            "Claim.",
            {"answer_id": "A1", "claims": [{"quote": "Wrong.", "start_char": 0, "end_char": 6, "label": "unknown", "fact_ids": [], "reason": ""}]},
            set(),
        )


def test_provider_timeout_is_preserved_as_answer_error():
    with pytest.raises(AnswerProcessingError, match="timed out"):
        ClaimAgent(FailingProvider(), FactStore([]), "v1").classify("A1", "cleartrail", "q", "Advice.")


def test_turn_limit_is_capped_and_gold_is_not_in_payload():
    with pytest.raises(ValueError):
        ClaimAgent(FailingProvider(), FactStore([]), "v1", max_turns=MAX_MODEL_TURNS + 1)

    class InspectingProvider:
        def __init__(self):
            self.messages = []

        def complete(self, messages, tools):
            self.messages.append(messages)
            return Response(json.dumps({"answer_id": "A1", "claims": []}))

    provider = InspectingProvider()
    ClaimAgent(provider, FactStore([]), "v1").classify("A1", "cleartrail", "q", "Advice.")
    serialized = json.dumps(provider.messages)
    assert "expected_label" not in serialized
    assert "development_gold" not in serialized
