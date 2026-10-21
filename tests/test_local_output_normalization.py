from speaker_agent.validation import validate_output


def test_accepts_single_claim_object_from_local_model():
    payload = {
        "answer_id": "A1",
        "claims": {
            "quote": "One.",
            "start_char": 0,
            "end_char": 4,
            "label": "unknown",
            "fact_ids": [],
            "reason": "",
        },
    }
    claims = validate_output("A1", "One.", payload, set())
    assert len(claims) == 1


def test_accepts_json_encoded_claim_list_from_local_model():
    payload = {
        "answer_id": "A1",
        "claims": '[{"quote":"One.","start_char":0,"end_char":4,"label":"unknown","fact_ids":[],"reason":""}]',
    }
    claims = validate_output("A1", "One.", payload, set())
    assert len(claims) == 1
