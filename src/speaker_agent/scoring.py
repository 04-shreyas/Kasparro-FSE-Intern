from __future__ import annotations

from collections import defaultdict
from typing import Any


def score_development(gold: list[dict[str, str]], predictions: dict[str, list[dict[str, Any]]], errors: dict[str, str]) -> dict[str, Any]:
    gold_by_id = {row["claim_id"]: row for row in gold}
    predicted: list[tuple[str, dict[str, Any]]] = [
        (answer_id, claim) for answer_id, claims in predictions.items() for claim in claims
    ]
    matched = 0
    correct = 0
    matched_ids: set[str] = set()
    wrong_ids: list[str] = []
    for answer_id, claim in predicted:
        claim_id = f"{answer_id}:{claim['start_char']}:{claim['end_char']}"
        row = gold_by_id.get(claim_id)
        if row is not None:
            matched += 1
            matched_ids.add(claim_id)
            if claim["label"] == row["expected_label"]:
                correct += 1
            else:
                wrong_ids.append(claim_id)
    gold_count, prediction_count = len(gold), len(predicted)

    def f1(p: int, r: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else 2 * p / denominator

    missed = sorted(set(gold_by_id) - matched_ids)
    extra = sorted(
        f"{answer_id}:{claim['start_char']}:{claim['end_char']}"
        for answer_id, claim in predicted
        if f"{answer_id}:{claim['start_char']}:{claim['end_char']}" not in gold_by_id
    )
    return {
        "gold_claims": gold_count,
        "accepted_predictions": prediction_count,
        "matched_spans": matched,
        "correct_labels": correct,
        "extraction_precision": matched / prediction_count if prediction_count else 0.0,
        "extraction_recall": matched / gold_count if gold_count else 0.0,
        "extraction_f1": f1(matched, matched, prediction_count + gold_count),
        "joint_precision": correct / prediction_count if prediction_count else 0.0,
        "joint_recall": correct / gold_count if gold_count else 0.0,
        "joint_f1": f1(correct, correct, prediction_count + gold_count),
        "gold_claim_coverage": correct / gold_count if gold_count else 0.0,
        "missed_gold_ids": missed,
        "unmatched_prediction_ids": extra,
        "matched_wrong_label_ids": sorted(wrong_ids),
        "answer_error_count": len(errors),
        "answer_errors": errors,
    }


def check_summary(predictions: dict[str, list[dict[str, Any]]], errors: dict[str, str]) -> dict[str, Any]:
    return {
        "gold_claims": None,
        "accepted_predictions": sum(len(x) for x in predictions.values()),
        "matched_spans": None,
        "correct_labels": None,
        "extraction_precision": None,
        "extraction_recall": None,
        "extraction_f1": None,
        "joint_precision": None,
        "joint_recall": None,
        "joint_f1": None,
        "gold_claim_coverage": None,
        "answer_error_count": len(errors),
        "answer_errors": errors,
    }
