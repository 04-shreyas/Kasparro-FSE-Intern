import pytest

from speaker_agent.scoring import score_development
from speaker_agent.validation import OutputValidationError, validate_output


def test_offsets_are_unicode_code_points_and_claim_id_is_derived():
    answer = "Advice. Café Mini costs INR 10."
    start = answer.index("Café")
    payload = {
        "answer_id": "A1",
        "claims": [{"quote": "Café Mini costs INR 10.", "start_char": 0, "end_char": 1, "label": "supported", "fact_ids": [], "reason": "It is listed."}],
    }
    claims = validate_output("A1", answer, payload, set())
    assert claims[0].start_char == start
    assert claims[0].claim_id == f"A1:{start}:{len(answer)}"


def test_rejects_duplicate_or_overlapping_spans():
    payload = {
        "answer_id": "A1",
        "claims": [
            {"quote": "One.", "start_char": 0, "end_char": 4, "label": "unknown", "fact_ids": [], "reason": ""},
            {"quote": "One.", "start_char": 0, "end_char": 4, "label": "unknown", "fact_ids": [], "reason": ""},
        ],
    }
    with pytest.raises(OutputValidationError):
        validate_output("A1", "One.", payload, set())


def test_rejects_invented_fact_and_bad_label():
    payload = {"answer_id": "A1", "claims": [{"quote": "One.", "label": "maybe", "fact_ids": ["fake"], "reason": ""}]}
    with pytest.raises(OutputValidationError):
        validate_output("A1", "One.", payload, {"real"})


def test_scorer_uses_all_predictions_and_gold():
    gold = [
        {"claim_id": "A1:0:4", "expected_label": "supported"},
        {"claim_id": "A1:5:9", "expected_label": "unknown"},
    ]
    predictions = {"A1": [{"start_char": 0, "end_char": 4, "label": "supported"}, {"start_char": 10, "end_char": 14, "label": "unknown"}]}
    metrics = score_development(gold, predictions, {"A2": "provider failed"})
    assert metrics["matched_spans"] == 1
    assert metrics["accepted_predictions"] == 2
    assert metrics["correct_labels"] == 1
    assert metrics["missed_gold_ids"] == ["A1:5:9"]
    assert metrics["answer_error_count"] == 1


def test_scorer_arithmetic_example():
    gold = [{"claim_id": f"A:{i}:x", "expected_label": "supported"} for i in range(10)]
    predictions = {"A": [{"start_char": i, "end_char": "x", "label": "supported"} for i in range(7)] + [{"start_char": i, "end_char": "x", "label": "unknown"} for i in range(7, 8)] + [{"start_char": i, "end_char": "x", "label": "supported"} for i in range(10, 12)]}
    metrics = score_development(gold, predictions, {})
    assert metrics["extraction_f1"] == pytest.approx(0.8)
    assert metrics["joint_f1"] == pytest.approx(0.7)
