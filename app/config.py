import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")
SAMPLES_DIR = "./data/samples"
TOP_K = int(os.getenv("TOP_K", "3"))
COLLECTION = "policies"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CHUNK_CHARS = 800

# If the closest chunk is farther than this (cosine distance), skip the LLM and decline.
MAX_DISTANCE = float(os.getenv("EVIDENCE_MAX_DISTANCE", "0.55"))
