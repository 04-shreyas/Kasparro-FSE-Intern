import json

import pytest

from speaker_agent.agent import AnswerProcessingError, ClaimAgent
from speaker_agent.data import load_facts
from speaker_agent.facts import FactStore


class Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class Response:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class Function:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class Call:
    def __init__(self, name, arguments):
        self.id = "call-1"
        self.function = Function(name, arguments)


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages, tools):
        self.messages.append(messages)
        return next(self.responses)


def make_agent(provider):
    return ClaimAgent(provider, FactStore(load_facts(__import__("pathlib").Path("brand_facts.csv"))), "v1")


def test_agent_requires_real_fact_tool_call():
    provider = FakeProvider(
        [
            Response(Message(tool_calls=[Call("get_brand_facts", json.dumps({"brand_id": "cleartrail"}))])),
            Response(Message('{"answer_id":"A1","claims":[]}')),
        ]
    )
    claims, _ = make_agent(provider).classify("A1", "cleartrail", "q", "Advice.")
    assert claims == []
    assert any(m.get("name") == "get_brand_facts" for m in provider.messages[1] if isinstance(m, dict))


def test_agent_rejects_illegal_tool():
    provider = FakeProvider([Response(Message(tool_calls=[Call("delete_data", "{}")]))])
    with pytest.raises(AnswerProcessingError):
        make_agent(provider).classify("A1", "cleartrail", "q", "Advice.")


def test_agent_preserves_invalid_provider_output_as_error():
    invalid = Response(Message('{"answer_id":"A1","claims":[{"quote":"Advice.","label":"supported","fact_ids":["fake"],"reason":""}]}'))
    provider = FakeProvider([invalid, invalid])
    with pytest.raises(AnswerProcessingError):
        make_agent(provider).classify("A1", "cleartrail", "q", "Advice.")


def test_ollama_content_tool_request_is_normalized():
    from speaker_agent.provider import OllamaProvider

    provider = OllamaProvider("unused", 1)
    response = provider._response_from_data(
        {"message": {"content": '{"name":"get_brand_facts","arguments":{"brand_id":"cleartrail"}}'}}
    )
    assert response.choices[0].message.tool_calls[0].function.name == "get_brand_facts"
