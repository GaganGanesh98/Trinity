"""
Minimal eval harness for RAG retrieval quality.

Run:
    python eval.py                     # local backend, uses eval_questions.json
    python eval.py --backend atlas     # MongoDB Atlas Vector Search backend
    python eval.py --file my_qs.json   # custom question set
    python eval.py --verbose           # print full answers
    python eval.py --retrieval-only    # score source recall, skip generation

Each question has:
    - question: the natural-language query
    - expected_sources: list of substrings that should appear in source titles
    - expected_answer_contains: list of substrings the answer should contain (optional)

Backends index the same corpus with the same chunking, embeddings, candidate
count and reranker, so scores are directly comparable. Latency is not strictly
comparable: `atlas` pays a network round trip per query that `local` does not.

--retrieval-only skips the LLM and scores source recall alone. Answer recall is
reported as n/a, and the latency column becomes retrieval time rather than
end-to-end time — which is the number worth comparing between backends anyway,
since end-to-end is dominated by local generation. A full run costs ~60s per
question; a retrieval-only run costs well under a second, so this is the mode to
use when iterating on retrieval and re-measuring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKENDS = ("local", "atlas")

DEFAULT_EVAL_FILE = Path(__file__).parent / "eval_questions.json"


def _get_backend(name: str, *, retrieval_only: bool = False):
    """Return (load_index, run_query) for the named backend.

    run_query returns {"answer": str, "sources": [...]}; in retrieval-only mode
    it returns {"sources": [...]} with no "answer" key.
    """
    if name == "local":
        from rag.indexer import load_index, query_with_sources, retrieve_sources
    elif name == "atlas":
        from rag.mongo_indexer import load_index, query_with_sources, retrieve_sources
    else:
        raise ValueError(f"Unknown backend {name!r}. Choose from {BACKENDS}.")
    return load_index, (retrieve_sources if retrieval_only else query_with_sources)


def run_eval(
    eval_file: Path,
    *,
    verbose: bool = False,
    backend: str = "local",
    retrieval_only: bool = False,
) -> dict:
    questions = json.loads(eval_file.read_text())
    load_index, run_query = _get_backend(backend, retrieval_only=retrieval_only)
    mode = "retrieval-only" if retrieval_only else "full (retrieval + generation)"
    print(f"Backend: {backend}  ·  Mode: {mode}")
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
        result = run_query(question)
        elapsed = time.time() - start

        answer = result.get("answer", "")
        sources = result["sources"]
        source_titles = [s["title"] for s in sources]

        if verbose:
            if not retrieval_only:
                print(f"\nAnswer ({elapsed:.1f}s):\n{answer}\n")
            print(f"Sources ({elapsed:.2f}s):")
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

        # Check answer content. Retrieval-only mode has no answer to check, so
        # those expectations are skipped rather than counted as misses — they
        # must not drag answer recall toward zero.
        answer_hits = 0
        if not retrieval_only:
            for exp in expected_answer:
                found = exp.lower() in answer.lower()
                status = "HIT" if found else "MISS"
                print(f"  Answer contains '{exp}': {status}")
                if found:
                    answer_hits += 1

        total_source_hits += source_hits
        total_source_expected += len(expected_sources)
        if not retrieval_only:
            total_answer_hits += answer_hits
            total_answer_expected += len(expected_answer)

        results.append({
            "question": question,
            "source_precision": source_hits / len(expected_sources) if expected_sources else 1.0,
            "answer_precision": (
                None if retrieval_only
                else (answer_hits / len(expected_answer) if expected_answer else 1.0)
            ),
            "latency_s": round(elapsed, 2),
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — backend: {backend} ({mode})")
    print(f"{'='*60}")
    src_recall = total_source_hits / total_source_expected if total_source_expected else 1.0
    print(f"Source recall:  {total_source_hits}/{total_source_expected} ({src_recall:.0%})")
    if retrieval_only:
        ans_recall = None
        print("Answer recall:  n/a (retrieval-only)")
    else:
        ans_recall = (
            total_answer_hits / total_answer_expected if total_answer_expected else 1.0
        )
        print(f"Answer recall:  {total_answer_hits}/{total_answer_expected} ({ans_recall:.0%})")
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0
    # Retrieval-only latency is the backend comparison worth making; the
    # end-to-end number is dominated by local generation. See README.
    label = "Avg retrieval" if retrieval_only else "Avg latency"
    print(f"{label}:  {avg_latency:.2f}s")

    return {
        "backend": backend,
        "mode": "retrieval_only" if retrieval_only else "full",
        "source_recall": round(src_recall, 4),
        "answer_recall": None if ans_recall is None else round(ans_recall, 4),
        "avg_latency_s": round(avg_latency, 2),
        "details": results,
    }


def _print_comparison(summaries: list[dict]) -> None:
    """Markdown table, ready to paste into the README."""
    print(f"\n{'='*60}")
    print("BACKEND COMPARISON")
    print(f"{'='*60}\n")
    retrieval_only = all(s.get("mode") == "retrieval_only" for s in summaries)
    latency_col = "Avg retrieval" if retrieval_only else "Avg latency"
    print(f"| Backend | Source recall | Answer recall | {latency_col} |")
    print("| --- | --- | --- | --- |")
    for s in summaries:
        ans = "n/a" if s["answer_recall"] is None else f"{s['answer_recall']:.0%}"
        print(
            f"| {s['backend']} | {s['source_recall']:.0%} "
            f"| {ans} | {s['avg_latency_s']:.2f}s |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG eval harness")
    parser.add_argument("--file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--backend",
        choices=[*BACKENDS, "all"],
        default="local",
        help="Retrieval backend to score. 'all' runs each in turn and prints a "
        "comparison table (default: local).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and score source recall only. Orders of magnitude "
        "faster (no LLM), and reports retrieval latency instead of end-to-end.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write the summary (or summaries) to FILE as JSON.",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Eval file not found: {args.file}")
        sys.exit(1)

    targets = list(BACKENDS) if args.backend == "all" else [args.backend]
    summaries = [
        run_eval(
            args.file,
            verbose=args.verbose,
            backend=b,
            retrieval_only=args.retrieval_only,
        )
        for b in targets
    ]

    if len(summaries) > 1:
        _print_comparison(summaries)

    if args.out:
        args.out.write_text(json.dumps(
            summaries if len(summaries) > 1 else summaries[0], indent=2
        ))
        print(f"\nWrote {args.out}")
