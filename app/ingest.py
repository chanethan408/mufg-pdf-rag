import hashlib
import shutil
from pathlib import Path

import pymupdf

from . import config, rag


class DocumentError(ValueError):
    pass


def extract_pages(source):
    """Return [(page_number, text)] from a PDF path or raw bytes."""
    try:
        if isinstance(source, bytes):
            doc = pymupdf.open(stream=source, filetype="pdf")
        else:
            doc = pymupdf.open(source)
        with doc:
            pages = [(i + 1, page.get_text().strip()) for i, page in enumerate(doc)]
    except Exception as e:
        raise DocumentError("could not read the PDF") from e
    if not any(text for _, text in pages):
        raise DocumentError("the PDF has no extractable text")
    return pages


def chunk_page(text, max_chars=config.CHUNK_CHARS):
    """Pack paragraphs into chunks of at most max_chars without splitting a paragraph."""
    chunks, current = [], ""
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def ingest_pdf(path):
    """Index one PDF. Returns "added", "updated" or "unchanged". Raises DocumentError for a bad PDF."""
    path = Path(path)
    doc_id = path.stem
    doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    col = rag.get_collection()

    old = col.get(where={"doc_id": doc_id})
    if old["ids"] and old["metadatas"][0]["doc_hash"] == doc_hash:
        return "unchanged"

    # build and embed everything first, so a failure leaves the old version in place
    ids, texts, metas = [], [], []
    for page, page_text in extract_pages(path):
        for piece in chunk_page(page_text):
            ids.append(f"{doc_id}-p{page}-c{len(ids)}")
            texts.append(piece)
            metas.append({"doc_id": doc_id, "filename": path.name, "page": page, "doc_hash": doc_hash})
    embeddings = rag.embed(texts)

    if old["ids"]:
        col.delete(ids=old["ids"])
    col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    return "updated" if old["ids"] else "added"


def ingest_all(rebuild=False):
    """Index every PDF in the PDF folder. Returns {filename: status or error message}."""
    if rebuild:
        try:
            rag.get_client().delete_collection(config.COLLECTION)
        except Exception:
            pass
    results = {}
    for pdf in sorted(Path(config.PDF_DIR).glob("*.pdf")):
        try:
            results[pdf.name] = ingest_pdf(pdf)
        except DocumentError as e:
            results[pdf.name] = f"failed: {e}"
    return results


def copy_samples():
    """Copy any sample PDF that is missing from the PDF folder."""
    Path(config.PDF_DIR).mkdir(parents=True, exist_ok=True)
    for src in Path(config.SAMPLES_DIR).glob("*.pdf"):
        dst = Path(config.PDF_DIR) / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)


def remove_document(doc_id):
    """Delete a document's chunks from the index. Returns False if it wasn't indexed."""
    col = rag.get_collection()
    ids = col.get(where={"doc_id": doc_id})["ids"]
    if ids:
        col.delete(ids=ids)
    return bool(ids)
