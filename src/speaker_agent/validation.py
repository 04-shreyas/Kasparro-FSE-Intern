from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

LABELS = {"supported", "contradicted", "unknown"}


class OutputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Claim:
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: list[str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputValidationError("Provider returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise OutputValidationError("Provider output must be a JSON object")
    return value


def _resolve_span(answer: str, raw: dict[str, Any]) -> tuple[int, int, str]:
    quote = raw.get("quote")
    if not isinstance(quote, str) or not quote:
        raise OutputValidationError("Claim quote must be a non-empty string")
    start = raw.get("start_char")
    end = raw.get("end_char")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(answer):
        if answer[start:end] == quote and quote == quote.strip():
            return start, end, quote
    occurrences: list[int] = []
    cursor = answer.find(quote)
    while cursor >= 0:
        occurrences.append(cursor)
        cursor = answer.find(quote, cursor + 1)
    if len(occurrences) != 1:
        raise OutputValidationError("Claim offsets do not match quote and quote is not uniquely locatable")
    start = occurrences[0]
    return start, start + len(quote), quote


def validate_output(
    answer_id: str,
    answer: str,
    payload: dict[str, Any],
    known_fact_ids: set[str],
) -> list[Claim]:
    if payload.get("answer_id") != answer_id:
        raise OutputValidationError("answer_id does not match the requested answer")
    raw_claims = payload.get("claims")
    if isinstance(raw_claims, str):
        try:
            raw_claims = json.loads(raw_claims)
        except json.JSONDecodeError as exc:
            raise OutputValidationError("claims must be a list") from exc
    if isinstance(raw_claims, dict):
        raw_claims = [raw_claims]
    if not isinstance(raw_claims, list):
        raise OutputValidationError("claims must be a list")
    claims: list[Claim] = []
    previous_end = -1
    seen_ids: set[str] = set()
    for raw in raw_claims:
        if not isinstance(raw, dict):
            raise OutputValidationError("Each claim must be an object")
        start, end, quote = _resolve_span(answer, raw)
        claim_id = f"{answer_id}:{start}:{end}"
        if claim_id in seen_ids or start < previous_end:
            raise OutputValidationError("Claims must be unique, non-overlapping, and ordered")
        label = raw.get("label")
        if label not in LABELS:
            raise OutputValidationError(f"Unsupported label: {label}")
        fact_ids = raw.get("fact_ids", [])
        if not isinstance(fact_ids, list) or any(not isinstance(x, str) for x in fact_ids):
            raise OutputValidationError("fact_ids must be a list of strings")
        if any(x not in known_fact_ids for x in fact_ids):
            raise OutputValidationError("Claim contains an invented fact ID")
        reason = raw.get("reason", "")
        if not isinstance(reason, str) or len(reason.split(". ")) > 2:
            raise OutputValidationError("reason must contain at most two sentences")
        claims.append(Claim(claim_id, start, end, quote, label, fact_ids, reason))
        seen_ids.add(claim_id)
        previous_end = end
    return claims
