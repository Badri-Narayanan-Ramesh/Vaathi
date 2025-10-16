from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

from core.ingest import load_pdf
from core.embeddings import build_index
from core.retriever import retrieve_for_question
from core.chains import (
    explain_page,
    answer_question,
    make_flashcards,
    make_quiz,
    make_cheatsheet,
)


def _nonempty(s: str | List | Dict | None) -> bool:
    if s is None:
        return False
    if isinstance(s, str):
        return len(s.strip()) > 0
    if isinstance(s, (list, dict)):
        return len(s) > 0
    return False


def main():
    ap = argparse.ArgumentParser(description="Full per-page suite in local mode: explain, answer (with citations), flashcards, quiz, cheatsheet")
    ap.add_argument("--pdf", type=str, required=True, help="Path to PDF")
    ap.add_argument("--out_dir", type=str, default="./out/local_suite", help="Output directory")
    ap.add_argument("--k", type=int, default=3, help="Top-k for retrieval")
    args = ap.parse_args()

    os.environ["APP_MODE"] = "local"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pdf(args.pdf)
    print(f"Loaded {len(pages)} pages from {args.pdf}")
    index = build_index(pages, index_dir=str(out_dir / "index"))

    summary_rows = []
    for p in pages:
        pid = int(p["page_id"])
        ctx = p["page_context"]
        page_dir = out_dir / f"page-{pid:02d}"
        page_dir.mkdir(exist_ok=True)

        # Explain
        explain = explain_page(ctx)
        (page_dir / "explain.txt").write_text(explain, encoding="utf-8")

        # Retrieval contexts for this page's question
        question = "Summarize this slide using the provided contexts and cite slides."
        hits = retrieve_for_question(index, question, k=args.k)
        ctx_texts = [h["text"] for h in hits] or [ctx]
        answer = answer_question(question, ctx_texts)
        (page_dir / "answer_with_citations.txt").write_text(answer, encoding="utf-8")
        (page_dir / "retrieval_hits.json").write_text(json.dumps(hits, indent=2), encoding="utf-8")

        # Flashcards
        cards = make_flashcards(ctx)
        (page_dir / "flashcards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")

        # Quiz
        quiz = make_quiz(ctx)
        (page_dir / "quiz.json").write_text(json.dumps(quiz, indent=2), encoding="utf-8")

        # Cheatsheet
        cheat = make_cheatsheet(ctx)
        (page_dir / "cheatsheet.txt").write_text(cheat, encoding="utf-8")

        summary_rows.append(
            {
                "page_id": pid,
                "explain_nonempty": _nonempty(explain),
                "answer_nonempty": _nonempty(answer),
                "flashcards_nonempty": _nonempty(cards),
                "quiz_nonempty": _nonempty(quiz),
                "cheatsheet_nonempty": _nonempty(cheat),
                "retrieval_count": len(hits),
            }
        )

    # Write summary
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    # Print a small PASS/FAIL table
    print("\nFunctionality summary (per page):")
    for r in summary_rows:
        print(
            f"Page {r['page_id']}: explain={'PASS' if r['explain_nonempty'] else 'FAIL'}, "
            f"answer={'PASS' if r['answer_nonempty'] else 'FAIL'}, "
            f"flashcards={'PASS' if r['flashcards_nonempty'] else 'FAIL'}, "
            f"quiz={'PASS' if r['quiz_nonempty'] else 'FAIL'}, "
            f"cheatsheet={'PASS' if r['cheatsheet_nonempty'] else 'FAIL'} (retrieval={r['retrieval_count']})"
        )
    print(f"\nOutputs saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
