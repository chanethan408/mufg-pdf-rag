import openai
import pytest
from fastapi.testclient import TestClient

from app import config, ingest, rag
from app.main import app

client = TestClient(app, raise_server_exceptions=False)
SECRET = "sk-secret-test-key"


@pytest.fixture
def indexed(env):
    ingest.ingest_all()
    return env


def test_retrieval_returns_citation_info(indexed):
    hits = rag.retrieve("minimum DSCR commercial loans", k=3)
    assert hits[0]["filename"] == "lending.pdf" and hits[0]["page"] >= 1 and "1.25x" in hits[0]["text"]


def test_empty_index_raises(env):
    with pytest.raises(rag.IndexNotReadyError):
        rag.retrieve("anything")


def test_far_away_question_declines_without_calling_llm(indexed, monkeypatch):
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.5)
    monkeypatch.setattr(rag, "generate", lambda *a: pytest.fail("LLM should not be called"))
    res = rag.answer_question("zebra quantum xylophone")
    assert res["abstained"] and res["abstain_reason"] == "no_relevant_evidence" and res["sources"] == []


class FakeOpenAI:
    reply = "INSUFFICIENT_EVIDENCE"
    error = None

    def __init__(self, api_key=None):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        if FakeOpenAI.error:
            raise FakeOpenAI.error
        msg = type("M", (), {"content": FakeOpenAI.reply})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    FakeOpenAI.reply, FakeOpenAI.error = "INSUFFICIENT_EVIDENCE", None
    return FakeOpenAI


def test_llm_declines(indexed, fake_llm):
    res = rag.answer_question("minimum DSCR commercial loans")
    assert res["abstained"] and res["abstain_reason"] == "llm_abstained" and res["sources"] == []


def test_llm_answer_comes_with_sources(indexed, fake_llm):
    fake_llm.reply = "The minimum DSCR is 1.25x [1]."
    res = rag.answer_question("minimum DSCR commercial loans")
    assert not res["abstained"] and "1.25x" in res["answer"] and res["sources"]


def test_llm_failure_gives_502_and_leaks_no_key(indexed, fake_llm):
    fake_llm.error = openai.OpenAIError(f"auth failed for {SECRET}")
    r = client.post("/ask", json={"question": "minimum DSCR commercial loans"})
    assert r.status_code == 502 and SECRET not in r.text


def test_missing_api_key_gives_502(indexed, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    assert client.post("/ask", json={"question": "minimum DSCR commercial loans"}).status_code == 502
