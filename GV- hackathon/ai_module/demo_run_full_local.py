from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.ingest import load_pdf
from core.chains import explain_page


def main():
    ap = argparse.ArgumentParser(description="Print full, untruncated explanations per page (local mode)")
    ap.add_argument("--pdf", type=str, required=True, help="Path to PDF")
    args = ap.parse_args()

    # Force local mode for this run
    os.environ["APP_MODE"] = "local"

    pdf_path = Path(args.pdf).resolve()
    print(f"Loading PDF: {pdf_path}")

    pages = load_pdf(str(pdf_path))
    print(f"Loaded {len(pages)} pages\n")

    for p in pages:
        pid = int(p["page_id"])
        ctx = p["page_context"]
        print("=" * 80)
        print(f"PAGE {pid} – Full Explanation")
        print("-" * 80)
        full = explain_page(ctx)
        print(full)
        print()


if __name__ == "__main__":
    main()
