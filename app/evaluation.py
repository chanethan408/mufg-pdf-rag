import json
import re
from pathlib import Path

from . import config, rag

DATASET = Path("data/eval/questions.json")
CITE = re.compile(r"\[(\d+)\]")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def load_dataset(path=DATASET):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def matches(chunk, ev):
    """A chunk matches an evidence label if doc and page agree and the phrase appears in its text."""
    return chunk["doc_id"] == ev["doc_id"] and chunk["page"] == ev["page"] and norm(ev["contains"]) in norm(chunk["text"])


def covered(chunks, evidence):
    return [any(matches(c, e) for c in chunks) for e in evidence]


def hit_at_k(chunks, evidence, k=3):
    return any(covered(chunks[:k], evidence))


def recall_at_k(chunks, evidence, k=3):
    found = covered(chunks[:k], evidence)
    return sum(found) / len(found)


def citation_scores(answer, sources, evidence):
    """precision: share of cited passages that are labeled evidence. recall: share of evidence that is cited."""
    idx = sorted({int(m) for m in CITE.findall(answer)})
    if not idx or not all(1 <= i <= len(sources) for i in idx):
        return {"has_valid_citations": False, "precision": 0.0, "recall": 0.0}
    cited = [sources[i - 1] for i in idx]
    precision = sum(any(matches(c, e) for e in evidence) for c in cited) / len(cited)
    recall = sum(any(matches(c, e) for c in cited) for e in evidence) / len(evidence)
    return {"has_valid_citations": True, "precision": precision, "recall": recall}


def must_include_ok(answer, must_include):
    return all(norm(m) in norm(answer) for m in must_include)


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def run_eval(dataset=None, k=3, retrieval_only=False):
    dataset = dataset if dataset is not None else load_dataset()
    records = []
    for q in dataset:
        chunks = rag.retrieve(q["question"], k)
        rec = {"id": q["id"], "type": q["type"], "question": q["question"],
               "expected_abstain": q["expected_abstain"], "best_score": round(chunks[0]["score"], 4)}
        if q["evidence"]:
            rec["hit_at_k"] = hit_at_k(chunks, q["evidence"], k)
            rec["recall_at_k"] = recall_at_k(chunks, q["evidence"], k)
        if not retrieval_only:
            res = rag.answer_question(q["question"])
            rec.update(answer=res["answer"], abstained=res["abstained"])
            if q["evidence"] and not res["abstained"]:
                rec["answer_correct"] = must_include_ok(res["answer"], q["must_include"])
                rec["citations"] = citation_scores(res["answer"], res["sources"], q["evidence"])
        records.append(rec)
    return {"config": {"k": k, "llm": None if retrieval_only else config.OPENAI_MODEL,
                       "max_distance": config.MAX_DISTANCE},
            "summary": summarize(records, retrieval_only), "records": records}


def summarize(records, retrieval_only):
    answerable = [r for r in records if not r["expected_abstain"]]
    unsupported = [r for r in records if r["expected_abstain"]]
    s = {"hit_at_k": mean([float(r["hit_at_k"]) for r in answerable]),
         "recall_at_k": mean([r["recall_at_k"] for r in answerable])}
    if retrieval_only:
        return s
    answered = [r for r in answerable if not r["abstained"]]
    s.update(
        false_abstention_rate=mean([float(r["abstained"]) for r in answerable]),
        abstention_rate_on_unsupported=mean([float(r["abstained"]) for r in unsupported]),
        answer_correct_rate=mean([float(r["answer_correct"]) for r in answered]),
        citation_precision=mean([r["citations"]["precision"] for r in answered]),
        citation_recall=mean([r["citations"]["recall"] for r in answered]),
    )
    return s
