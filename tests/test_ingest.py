from app import ingest, rag
from tests.helpers import fake_embed, make_pdf


def all_ids():
    return rag.get_collection().get()["ids"]


def test_ingest_twice_changes_nothing(env):
    assert set(ingest.ingest_all().values()) == {"added"}
    ids = all_ids()
    assert set(ingest.ingest_all().values()) == {"unchanged"}
    assert sorted(all_ids()) == sorted(ids)


def test_update_replaces_old_content(env):
    ingest.ingest_all()
    make_pdf(env / "lending.pdf", ["The minimum DSCR for commercial loans is 1.40x."])
    assert ingest.ingest_all() == {"docs.pdf": "unchanged", "lending.pdf": "updated"}
    text = " ".join(h["text"] for h in rag.retrieve("minimum DSCR commercial loans", k=5))
    assert "1.40x" in text and "1.25x" not in text


def test_bad_pdfs_fail_but_good_ones_still_index(env):
    (env / "corrupt.pdf").write_bytes(b"not a pdf")
    make_pdf(env / "blank.pdf", [""])
    results = ingest.ingest_all()
    assert results["corrupt.pdf"].startswith("failed") and results["blank.pdf"].startswith("failed")
    assert results["lending.pdf"] == "added"


def test_embedding_failure_keeps_old_version(env, monkeypatch):
    ingest.ingest_all()
    make_pdf(env / "lending.pdf", ["The minimum DSCR for commercial loans is 1.50x."])

    def boom(texts):
        raise RuntimeError("embedder down")
    monkeypatch.setattr(rag, "embed", boom)
    try:
        ingest.ingest_pdf(env / "lending.pdf")
    except RuntimeError:
        pass
    monkeypatch.setattr(rag, "embed", fake_embed)
    assert any("1.25x" in h["text"] for h in rag.retrieve("minimum DSCR", k=3))


def test_chunk_metadata(env):
    ingest.ingest_all()
    data = rag.get_collection().get(include=["metadatas", "documents"])
    for meta, doc in zip(data["metadatas"], data["documents"]):
        assert doc.strip() and meta["doc_id"] and meta["filename"].endswith(".pdf") and meta["page"] >= 1
