import pytest
from fastapi.testclient import TestClient

from app import config, ingest, rag
from app.main import app
from tests.helpers import make_pdf

client = TestClient(app, raise_server_exceptions=False)


def pdf_bytes(tmp_path, text, name="up.pdf"):
    p = tmp_path / name
    make_pdf(p, [text])
    return p.read_bytes()


def put(name, data):
    return client.put(f"/documents/{name}", content=data)


def test_upload_new_document_is_searchable(env, tmp_path):
    ingest.ingest_all()
    r = put("travel_policy.pdf", pdf_bytes(tmp_path, "Business class flights need VP approval above 6 hours."))
    assert r.status_code == 200 and r.json()["status"] == "added"
    assert (env / "travel_policy.pdf").exists()
    assert "VP approval" in rag.retrieve("business class flights approval", k=1)[0]["text"]
    listed = {d["filename"]: d for d in client.get("/documents").json()}
    assert listed["travel_policy.pdf"]["indexed"] is True


def test_reupload_unchanged_then_modified(env, tmp_path):
    ingest.ingest_all()
    v1 = pdf_bytes(tmp_path, "The flight limit is 6 hours.")
    assert put("travel.pdf", v1).json()["status"] == "added"
    assert put("travel.pdf", v1).json()["status"] == "unchanged"
    assert put("travel.pdf", pdf_bytes(tmp_path, "The flight limit is 8 hours.", "v2.pdf")).json()["status"] == "updated"
    text = " ".join(h["text"] for h in rag.retrieve("flight limit hours", k=5))
    assert "8 hours" in text and "6 hours" not in text


def test_bad_upload_does_not_replace_good_document(env, tmp_path):
    ingest.ingest_all()
    good = pdf_bytes(tmp_path, "The flight limit is 6 hours.")
    put("travel.pdf", good)
    assert put("travel.pdf", b"not a pdf at all").status_code == 400
    assert put("travel.pdf", b"%PDF-1.4 truncated garbage").status_code == 422
    assert (env / "travel.pdf").read_bytes() == good
    assert any("6 hours" in h["text"] for h in rag.retrieve("flight limit", k=3))


@pytest.mark.parametrize("name", ["notes.txt", "a$b.pdf", "..\\evil.pdf", "x" * 200 + ".pdf", ".hidden.pdf"])
def test_invalid_filenames_rejected(env, name):
    assert put(name, b"%PDF-1.4").status_code == 400


def test_upload_size_limit(env, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 50)
    assert put("big.pdf", b"%PDF-" + b"x" * 500).status_code == 413


def test_remove_document(env):
    ingest.ingest_all()
    assert client.delete("/documents/docs.pdf").status_code == 200
    assert not (env / "docs.pdf").exists()
    assert all(h["doc_id"] != "docs" for h in rag.retrieve("loan files retained years", k=5))
    assert client.delete("/documents/docs.pdf").status_code == 404
    assert client.delete("/documents/bad$name.pdf").status_code == 400


def test_removing_everything_gives_clear_errors(env):
    ingest.ingest_all()
    client.delete("/documents/docs.pdf")
    client.delete("/documents/lending.pdf")
    assert client.get("/health").status_code == 503
    assert client.post("/ask", json={"question": "anything"}).status_code == 503


def test_removed_sample_comes_back_on_restart(env, tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    make_pdf(samples / "sample_a.pdf", ["Default policy A: the review period is 12 months."])
    monkeypatch.setattr(config, "SAMPLES_DIR", str(samples))

    ingest.copy_samples()
    ingest.ingest_all()
    client.delete("/documents/sample_a.pdf")
    assert not (env / "sample_a.pdf").exists()
    ingest.copy_samples()
    ingest.ingest_all()
    assert any(h["doc_id"] == "sample_a" for h in rag.retrieve("review period months", k=3))


def test_sample_pdfs_exist_in_repo():
    from pathlib import Path
    names = {p.name for p in Path("data/samples").glob("*.pdf")}
    assert names >= {"lending_requirements.pdf", "credit_risk_review.pdf", "loan_documentation.pdf"}
