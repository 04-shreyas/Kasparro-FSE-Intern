from pathlib import Path

from speaker_agent.data import load_answers, load_facts
from speaker_agent.facts import FactLookupError, FactStore


ROOT = Path(__file__).parents[1]


def test_csv_parser_preserves_records_and_embedded_text():
    answers = load_answers(ROOT / "answers.csv")
    assert len(answers) == 50
    assert {a.split for a in answers} == {"development", "check"}
    assert any("\n" in a.ai_answer for a in answers)


def test_fact_tool_is_read_only_and_rejects_unknown_brand():
    store = FactStore(load_facts(ROOT / "brand_facts.csv"))
    result = store.get_brand_facts("cleartrail")
    assert result["fact_version"] == "speaker-catalogue-v4"
    assert len(result["facts"]) == 44
    try:
        store.get_brand_facts("not-a-brand")
    except FactLookupError:
        pass
    else:
        raise AssertionError("unknown brands must be rejected")
