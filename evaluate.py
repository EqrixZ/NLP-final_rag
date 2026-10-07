"""Evaluate the RAG pipeline on ``test_questions.csv``.

Runs every question through the SAME retrieval + generation code as the app
(``rag.pipeline``) and reports:

* retrieval hit rate — did the expected source file appear in the top-k?
* no-info accuracy  — do unanswerable questions yield ``status == "no_info"``?
* false no-info     — do answerable questions get ``status == "no_info"``?

Usage::

    export GROQ_API_KEY=...            # read from the environment (never hard-coded)
    python evaluate.py                 # full evaluation (retrieval + Groq generation)
    python evaluate.py --retrieval-only   # no API key needed

Results are written to ``eval_results.csv`` (UTF-8 with BOM for Excel).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from rag.config import DEFAULT_TOP_K, GROQ_MODEL, PROJECT_ROOT, SIMILARITY_THRESHOLD
from rag.index import load_embedder
from rag.llm import GroqChatModel, LLMError
from rag.pipeline import RAGPipeline, build_index


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_questions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "test_questions.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_results.csv")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM (no API key needed)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between LLM calls (rate limits)")
    args = parser.parse_args()

    llm = None
    if not args.retrieval_only:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY is not set. Export it, or run with --retrieval-only.", file=sys.stderr)
            return 1
        llm = GroqChatModel(api_key=api_key, model=os.environ.get("GROQ_MODEL", GROQ_MODEL))

    questions = load_questions(args.questions)
    print(f"Building index ... ({len(questions)} questions, top_k={args.top_k}, threshold={args.threshold})")
    index = build_index(load_embedder())
    pipeline = RAGPipeline(index, llm, threshold=args.threshold)
    print(f"Index ready: {len(index)} chunks\n")

    rows: list[dict[str, object]] = []
    for q in questions:
        answerable = parse_bool(q["answerable"])
        expected = {s.strip() for s in q.get("source_file", "").split(";") if s.strip()}

        # Raw top-k (no threshold) to measure retrieval quality independently of the cut-off.
        raw_hits = index.search(q["question"], top_k=args.top_k, threshold=0.0)
        retrieved_files = [h.chunk.source_file for h in raw_hits]
        hit = bool(expected & set(retrieved_files)) if answerable else None

        try:
            result = pipeline.answer(q["question"], top_k=args.top_k)
        except LLMError as exc:  # defensive; the pipeline already converts these
            print(f"[{q['id']}] LLM error: {exc}", file=sys.stderr)
            continue

        refused = result.status == "no_info"
        # Without an LLM, no-info is only known when the threshold rejected every chunk (layer 1).
        measured = llm is not None or not result.sources
        row = {
            "id": q["id"],
            "question": q["question"],
            "language": q["language"],
            "answerable": answerable,
            "expected_source": ";".join(sorted(expected)),
            "retrieved_files": ";".join(dict.fromkeys(retrieved_files)),
            "top_score": round(raw_hits[0].score, 4) if raw_hits else "",
            "retrieval_hit": hit,
            "chunks_above_threshold": len(result.sources),
            "status": result.status if measured else "",
            "no_info_layer": ("retrieval" if not result.sources else "llm") if measured and refused else "",
            "refused": refused if measured else "",
            "refusal_correct": refused if measured and not answerable else "",
            "false_refusal": refused if measured and answerable else "",
            "cited": ";".join(str(n) for n in sorted(result.cited_ranks)),
            "rewritten_query": result.rewritten_query,
            "suggestions": " | ".join(result.suggestions),
            "answer": result.answer if measured else "",
            "error": result.error or "",
        }
        rows.append(row)
        status = "HIT " if hit else ("MISS" if hit is False else "N/A ")
        print(f"[{q['id']:>3}] {status} top={row['top_score']} status={row['status'] or '-':<8} {q['question'][:60]}")
        if llm:
            time.sleep(args.delay)

    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    hits = sum(1 for r in answerable_rows if r["retrieval_hit"])
    print("\n===== Summary =====")
    print(f"Retrieval hit rate @k={args.top_k} (answerable): {pct(hits, len(answerable_rows))}")
    if llm:
        correct_refusals = sum(1 for r in unanswerable_rows if r["refusal_correct"] is True)
        false_refusals = sum(1 for r in answerable_rows if r["false_refusal"] is True)
        errors = sum(1 for r in rows if r["error"])
        print(f"Unanswerable -> status == no_info:     {pct(correct_refusals, len(unanswerable_rows))}")
        print(f"Answerable wrongly -> no_info:         {pct(false_refusals, len(answerable_rows))}")
        print(f"Answers with >=1 citation:             {pct(sum(1 for r in answerable_rows if r['cited']), len(answerable_rows))}")
        print(f"LLM errors:                            {errors}")
    else:
        threshold_refusals = sum(1 for r in unanswerable_rows if r["chunks_above_threshold"] == 0)
        print(f"Unanswerable -> no_info by threshold alone (layer 1): {pct(threshold_refusals, len(unanswerable_rows))}")
        print("(Layer-2 / LLM no-info not measured: run without --retrieval-only and with GROQ_API_KEY set.)")
    print(f"\nSaved per-question results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
