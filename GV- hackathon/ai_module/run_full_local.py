from __future__ import annotations

import argparse
import os
from pathlib import Path
from collections import Counter
import re

from core.ingest import load_pdf
from core.chains import explain_page, answer_question, make_cheatsheet


STOPWORDS = {
    "the","and","a","an","of","to","in","on","for","with","as","by","at","is","are","was","were","be","been","or","that","this","these","those","it","its","from","into","over","under","about","than","then","so","such","not","no","we","you","they","he","she","his","her","their","our","your","but","if","because","while","during","per","i","ii","iii","iv","v",
}


def top_keywords(text: str, k: int = 12) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(k)]


def coverage_score(answer: str, page_text: str, kws: list[str]) -> float:
    if not kws:
        return 0.0
    ans_l = answer.lower()
    hit = sum(1 for w in kws if w in ans_l)
    return hit / len(kws)


def main():
    ap = argparse.ArgumentParser(description="Generate full local outputs per page and verify vs PDF content")
    ap.add_argument("--pdf", type=str, required=True, help="Path to PDF to ingest")
    ap.add_argument("--out_dir", type=str, default="./out/local_full", help="Output directory for text files")
    args = ap.parse_args()

    os.environ["APP_MODE"] = os.getenv("APP_MODE", "local")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pdf(args.pdf)
    print(f"Loaded {len(pages)} pages from {args.pdf}")

    report_lines = []
    for p in pages:
        pid = int(p["page_id"])
        ctx = p["page_context"]

        # Full, untruncated chains
        exp = explain_page(ctx)
        ans = answer_question("Summarize this slide using the provided context only.", [ctx])

        # Save outputs
        (out_dir / f"page-{pid:02d}-explain.txt").write_text(exp, encoding="utf-8")
        (out_dir / f"page-{pid:02d}-answer.txt").write_text(ans, encoding="utf-8")

        # Simple verification vs page content
        kws = top_keywords(ctx, k=12)
        cov = coverage_score(ans, ctx, kws)
        report_lines.append(
            f"Page {pid}: coverage={cov:.2f} matched_keywords={[w for w in kws if w in ans.lower()]}"
        )

    # Overall summaries
    combined_ctx = "\n\n".join(p.get("page_context", "") for p in pages)
    # Cap combined context to a reasonable size for local LLMs
    if len(combined_ctx) > 15000:
        combined_ctx = combined_ctx[:15000]

    overall_cheat = make_cheatsheet(combined_ctx)
    (out_dir / "overall_cheatsheet.txt").write_text(overall_cheat, encoding="utf-8")

    overall_explain = explain_page(combined_ctx)
    (out_dir / "overall_explain.txt").write_text(overall_explain, encoding="utf-8")

    (out_dir / "verification_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\nVerification summary:")
    print("\n".join(report_lines))
    print("\nOverall document summaries saved as 'overall_cheatsheet.txt' and 'overall_explain.txt'.")
    print(f"Full outputs saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
