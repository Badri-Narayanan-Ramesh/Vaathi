from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.ingest import load_pdf
from core.embeddings import build_index
from core.retriever import retrieve_for_question
from core.chains import explain_page, answer_question, make_flashcards, make_quiz, make_cheatsheet


def main():
    ap = argparse.ArgumentParser(description="AI Tutor backbone demo run")
    ap.add_argument("--pdf", type=str, default=str(Path(__file__).resolve().parents[1] / "test_pdf.pdf"), help="Path to PDF to ingest")
    ap.add_argument("--index_dir", type=str, default="./data/index", help="Chroma index directory")
    ap.add_argument("--k", type=int, default=3, help="Top-k retrieval")
    args = ap.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    print(f"Loading PDF: {pdf_path}")
    pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages")

    index = build_index(pages, index_dir=args.index_dir)
    print("Index built.")

    # Explain first page
    ep = explain_page(pages[0]["page_context"]) if pages else ""
    print("\n=== Explain Page 1 (truncated) ===\n", ep[:600])

    # Retrieval + answer
    hits = retrieve_for_question(index, "What is on the first slide?", k=args.k)
    print("\nTop hits:", [{"page_id": h["page_id"], "score": round(float(h.get("score", 0.0)), 4)} for h in hits])
    ctx_texts = [h["text"] for h in hits]
    ans = answer_question("Summarize the first slide.", ctx_texts)
    print("\n=== Answer with Citations (truncated) ===\n", ans[:800])

    # Study aids
    cards = make_flashcards(pages[0]["page_context"]) if pages else []
    quiz = make_quiz(pages[0]["page_context"]) if pages else []
    cheat = make_cheatsheet(pages[0]["page_context"]) if pages else ""
    print("\n=== Flashcards (up to 2) ===\n", cards[:2])
    print("\n=== Quiz (first item) ===\n", quiz[:1])
    print("\n=== Cheatsheet (truncated) ===\n", cheat[:600])


if __name__ == "__main__":
    main()
