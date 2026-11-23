import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pymupdf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from starlette.concurrency import run_in_threadpool

from . import config, ingest, rag


@asynccontextmanager
async def lifespan(_):
    ingest.copy_samples()
    await run_in_threadpool(ingest.ingest_all)
    yield


app = FastAPI(title="Credit Policy Intelligence API", lifespan=lifespan)
app.mount("/pdfs", StaticFiles(directory=config.PDF_DIR, check_dir=False), name="pdfs")

FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,100}\.pdf", re.IGNORECASE)


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


@app.exception_handler(rag.IndexNotReadyError)
async def index_not_ready(_: Request, exc: rag.IndexNotReadyError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(rag.LLMError)
async def llm_error(_: Request, exc: rag.LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    chunks = rag.index_count()
    return JSONResponse(status_code=200 if chunks else 503,
                        content={"status": "ok" if chunks else "empty", "index_ready": chunks > 0,
                                 "indexed_chunks": chunks})


@app.post("/ask")
def ask(req: AskRequest):
    return rag.answer_question(req.question)


@app.get("/documents")
def list_documents():
    indexed = {m["doc_id"] for m in rag.get_collection().get(include=["metadatas"])["metadatas"]}
    docs = []
    for pdf in sorted(Path(config.PDF_DIR).glob("*.pdf")):
        try:
            with pymupdf.open(pdf) as d:
                pages = len(d)
        except Exception:
            continue
        docs.append({"filename": pdf.name, "title": pdf.stem.replace("_", " ").title(),
                     "pages": pages, "indexed": pdf.stem in indexed})
    return docs


@app.put("/documents/{filename}")
async def upload_document(filename: str, request: Request):
    if not FILENAME_RE.fullmatch(filename):
        raise HTTPException(400, "Use letters, numbers, spaces, dots, dashes or underscores, ending in .pdf.")
    data = await request.body()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Maximum upload size is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "That file is not a PDF.")

    # check the PDF is readable before it can replace an existing file
    try:
        ingest.extract_pages(data)
    except ingest.DocumentError as e:
        raise HTTPException(422, f"Could not index this PDF: {e}.")

    path = Path(config.PDF_DIR) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    status = await run_in_threadpool(ingest.ingest_pdf, path)
    return {"filename": filename, "status": status}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    if not FILENAME_RE.fullmatch(filename):
        raise HTTPException(400, "Invalid filename.")
    path = Path(config.PDF_DIR) / filename
    was_indexed = ingest.remove_document(path.stem)
    had_file = path.exists()
    path.unlink(missing_ok=True)
    if not was_indexed and not had_file:
        raise HTTPException(404, "No such document.")
    return {"filename": filename, "removed": True}
