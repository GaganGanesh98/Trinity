"""
Minimal eval harness for RAG retrieval quality.

Run:
    python eval.py                     # uses eval_questions.json
    python eval.py --file my_qs.json   # custom question set
    python eval.py --verbose           # print full answers

Each question has:
    - question: the natural-language query
    - expected_sources: list of substrings that should appear in source titles
    - expected_answer_contains: list of substrings the answer should contain (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rag.indexer import load_index, query_with_sources

DEFAULT_EVAL_FILE = Path(__file__).parent / "eval_questions.json"


def run_eval(eval_file: Path, *, verbose: bool = False) -> dict:
    questions = json.loads(eval_file.read_text())
    load_index()

    results = []
    total_source_hits = 0
    total_source_expected = 0
    total_answer_hits = 0
    total_answer_expected = 0

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected_sources = q.get("expected_sources", [])
        expected_answer = q.get("expected_answer_contains", [])

        print(f"\n{'='*60}")
        print(f"Q{i}: {question}")

        start = time.time()
        result = query_with_sources(question)
        elapsed = time.time() - start

        answer = result["answer"]
        sources = result["sources"]
        source_titles = [s["title"] for s in sources]

        if verbose:
            print(f"\nAnswer ({elapsed:.1f}s):\n{answer}\n")
            print("Sources:")
            for s in sources:
                print(f"  - {s['title']} (score: {s['score']})")

        # Check source retrieval
        source_hits = 0
        for exp in expected_sources:
            found = any(exp.lower() in t.lower() for t in source_titles)
            status = "HIT" if found else "MISS"
            print(f"  Source '{exp}': {status}")
            if found:
                source_hits += 1

        # Check answer content
        answer_hits = 0
        for exp in expected_answer:
            found = exp.lower() in answer.lower()
            status = "HIT" if found else "MISS"
            print(f"  Answer contains '{exp}': {status}")
            if found:
                answer_hits += 1

        total_source_hits += source_hits
        total_source_expected += len(expected_sources)
        total_answer_hits += answer_hits
        total_answer_expected += len(expected_answer)

        results.append({
            "question": question,
            "source_precision": source_hits / len(expected_sources) if expected_sources else 1.0,
            "answer_precision": answer_hits / len(expected_answer) if expected_answer else 1.0,
            "latency_s": round(elapsed, 2),
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    src_recall = total_source_hits / total_source_expected if total_source_expected else 1.0
    ans_recall = total_answer_hits / total_answer_expected if total_answer_expected else 1.0
    print(f"Source recall:  {total_source_hits}/{total_source_expected} ({src_recall:.0%})")
    print(f"Answer recall:  {total_answer_hits}/{total_answer_expected} ({ans_recall:.0%})")
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0
    print(f"Avg latency:   {avg_latency:.1f}s")

    return {
        "source_recall": round(src_recall, 4),
        "answer_recall": round(ans_recall, 4),
        "avg_latency_s": round(avg_latency, 2),
        "details": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG eval harness")
    parser.add_argument("--file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Eval file not found: {args.file}")
        sys.exit(1)

    run_eval(args.file, verbose=args.verbose)
