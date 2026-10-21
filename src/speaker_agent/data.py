from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FACT_VERSION = "speaker-catalogue-v4"
DATASET_VERSION = "long-brand-answers-50-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"


@dataclass(frozen=True)
class Answer:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


@dataclass(frozen=True)
class Fact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: str
    value_type: str
    unit: str
    condition: str
    valid_from: str
    valid_to: str

    def as_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_answers(path: Path) -> list[Answer]:
    return [Answer(**row) for row in _read(path)]


def load_facts(path: Path) -> list[Fact]:
    return [Fact(**row) for row in _read(path)]


def load_gold(path: Path) -> list[dict[str, str]]:
    return _read(path)


def file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
