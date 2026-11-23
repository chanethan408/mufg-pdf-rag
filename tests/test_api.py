from fastapi.testclient import TestClient

from app import rag
from app.main import app

client = TestClient(app, raise_server_exceptions=False)

FAKE = {"question": "q", "answer": "DSCR is 1.25x [1]", "abstained": False, "abstain_reason": None,
        "best_score": 0.1, "sources": [{"filename": "d.pdf", "page": 1, "score": 0.1, "text": "t"}]}


def test_ui_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "Credit Policy Assistant" in r.text


def test_documents_list(env):
    docs = client.get("/documents").json()
    assert {d["filename"] for d in docs} == {"lending.pdf", "docs.pdf"}
    assert all(d["indexed"] is False for d in docs)


def test_ask(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda q: FAKE)
    r = client.post("/ask", json={"question": "min DSCR?"})
    assert r.status_code == 200 and r.json()["sources"][0]["page"] == 1


def test_bad_questions_rejected():
    for body in ({"question": ""}, {"question": "   "}, {}, {"question": 123}):
        assert client.post("/ask", json=body).status_code == 422


def raises(error):
    def fn(q):
        raise error
    return fn


def test_llm_error(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", raises(rag.LLMError("failed")))
    assert client.post("/ask", json={"question": "x"}).status_code == 502


def test_index_not_ready(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", raises(rag.IndexNotReadyError("empty")))
    assert client.post("/ask", json={"question": "x"}).status_code == 503


def test_unexpected_error_hides_details(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", raises(ValueError("secret sk-123")))
    r = client.post("/ask", json={"question": "x"})
    assert r.status_code == 500 and "sk-123" not in r.text


def test_health(monkeypatch):
    monkeypatch.setattr(rag, "index_count", lambda: 5)
    assert client.get("/health").status_code == 200
    monkeypatch.setattr(rag, "index_count", lambda: 0)
    assert client.get("/health").status_code == 503
