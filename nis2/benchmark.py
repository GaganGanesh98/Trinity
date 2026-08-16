"""
Benchmark assessment models against hand-labelled ground truth.

    python -m nis2.benchmark                          # default model
    python -m nis2.benchmark --models llama3.2 qwen2.5:7b
    python -m nis2.benchmark --out bench.json

Finding quality is the binding constraint on whether the report is usable, and
it is model-dependent — so it gets measured the same way retrieval does, rather
than judged by reading a couple of outputs.

Three things are scored separately, because a model can be right for the wrong
reason:

- **status accuracy** — did it reach the correct verdict?
- **gap recall** — of the checkpoints that are genuinely deficient, how many did
  it decline to mark ADDRESSED? A false clean bill of health is the costly
  error here: telling someone they are covered when they are not.
- **rationale recall** — for labels carrying `must_mention`, did the rationale
  actually raise the specific omission (e.g. the Art. 23 deadlines)?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nis2.assessor import assess
from nis2.schema import Status

GROUND_TRUTH = Path(__file__).parent / "ground_truth.json"

# Statuses that assert the document does NOT fully meet the obligation.
_GAP_STATUSES = {Status.PARTIAL, Status.NOT_ADDRESSED, Status.UNCLEAR}


def run(model: str, truth: dict) -> dict:
    doc_path = Path(truth["document"])
    data = doc_path.read_bytes()
    labels = truth["labels"]

    start = time.time()
    report = assess(data, doc_path.name, model=model)
    elapsed = time.time() - start

    rows, status_hits = [], 0
    gap_total = gap_caught = 0
    mention_total = mention_hits = 0

    for f in report.findings:
        exp = labels.get(f.checkpoint_id)
        if not exp:
            continue
        expected = exp["status"]
        correct = f.status.value == expected
        status_hits += correct

        # A genuine deficiency must not come back as ADDRESSED.
        if expected in ("PARTIAL", "NOT_ADDRESSED"):
            gap_total += 1
            gap_caught += f.status in _GAP_STATUSES

        # Did the rationale name the specific omission we care about?
        mentions = [m.lower() for m in exp.get("must_mention", [])]
        mention_ok = None
        if mentions:
            mention_total += 1
            mention_ok = any(m in f.rationale.lower() for m in mentions)
            mention_hits += mention_ok

        rows.append({
            "checkpoint": f.checkpoint_id,
            "domain": f.domain,
            "expected": expected,
            "got": f.status.value,
            "correct": correct,
            "quoted": bool(f.excerpt),
            "mention_ok": mention_ok,
        })

    n = len(rows) or 1
    return {
        "model": model,
        "status_accuracy": round(status_hits / n, 3),
        "gap_recall": round(gap_caught / gap_total, 3) if gap_total else None,
        "rationale_recall": round(mention_hits / mention_total, 3) if mention_total else None,
        "quoted_pct": round(sum(r["quoted"] for r in rows) / n, 3),
        "elapsed_s": round(elapsed, 1),
        "coverage_pct_reported": report.coverage_pct,
        "rows": rows,
    }


def _print(res: dict) -> None:
    print(f"\n{'='*66}\n{res['model']}  ({res['elapsed_s']}s)\n{'='*66}")
    print(f"{'checkpoint':<11} {'expected':<15} {'got':<15} {'':<3} quote")
    for r in res["rows"]:
        mark = "ok " if r["correct"] else "MISS"
        q = "yes" if r["quoted"] else "—"
        note = "" if r["mention_ok"] is None else ("  [omission named]" if r["mention_ok"] else "  [MISSED KEY OMISSION]")
        print(f"{r['checkpoint']:<11} {r['expected']:<15} {r['got']:<15} {mark:<4} {q}{note}")
    print(f"\nstatus accuracy   {res['status_accuracy']:.0%}")
    print(f"gap recall        {res['gap_recall']:.0%}" if res["gap_recall"] is not None else "")
    print(f"rationale recall  {res['rationale_recall']:.0%}" if res["rationale_recall"] is not None else "")
    print(f"findings quoted   {res['quoted_pct']:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark NIS2 assessment models")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Ollama models to compare (default: config.NIS2_LLM_MODEL)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    truth = json.loads(GROUND_TRUTH.read_text())
    models = args.models or [__import__("config").NIS2_LLM_MODEL]

    results = []
    for m in models:
        res = run(m, truth)
        _print(res)
        results.append(res)

    if len(results) > 1:
        print(f"\n{'='*66}\nCOMPARISON\n{'='*66}\n")
        print("| Model | Status accuracy | Gap recall | Rationale recall | Time |")
        print("| --- | --- | --- | --- | --- |")
        for r in results:
            gr = f"{r['gap_recall']:.0%}" if r["gap_recall"] is not None else "n/a"
            rr = f"{r['rationale_recall']:.0%}" if r["rationale_recall"] is not None else "n/a"
            print(f"| {r['model']} | {r['status_accuracy']:.0%} | {gr} | {rr} | {r['elapsed_s']:.0f}s |")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")
