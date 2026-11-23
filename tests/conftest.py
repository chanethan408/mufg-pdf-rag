import pytest

from app import config, rag
from tests.helpers import fake_embed, make_pdf


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp PDF and Chroma folders with fake embeddings, so no model or API is needed."""
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    monkeypatch.setattr(config, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "PDF_DIR", str(pdfs))
    monkeypatch.setattr(config, "MAX_DISTANCE", 10)  # tests set this when needed
    monkeypatch.setattr(rag, "embed", fake_embed)
    make_pdf(pdfs / "lending.pdf", ["The minimum DSCR for commercial loans is 1.25x.\n\nLoans need 24 months history."])
    make_pdf(pdfs / "docs.pdf", ["Loan files are retained for 7 years after maturity."])
    return pdfs
