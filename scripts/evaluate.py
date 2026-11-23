import argparse
import json
from pathlib import Path

from app import evaluation

ap = argparse.ArgumentParser()
ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM (free)")
ap.add_argument("--out", default="data/eval/results.json")
args = ap.parse_args()

result = evaluation.run_eval(retrieval_only=args.retrieval_only)
Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

for r in result["records"]:
    print(r["id"], r["type"], "best =", r["best_score"], "hit =", r.get("hit_at_k"), "abstained =", r.get("abstained"))
print("\nSummary")
for key, value in result["summary"].items():
    print(f"  {key}: {value}")
print("\nFull results:", args.out)
