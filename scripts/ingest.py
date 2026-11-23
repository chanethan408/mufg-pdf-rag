import argparse

from app import ingest, rag

ap = argparse.ArgumentParser()
ap.add_argument("--rebuild", action="store_true", help="drop the index and re-ingest everything")
args = ap.parse_args()

ingest.copy_samples()
for name, status in ingest.ingest_all(rebuild=args.rebuild).items():
    print(f"{status:<10} {name}")
print("Chunks indexed:", rag.index_count())
