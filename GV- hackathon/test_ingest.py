"""
Tester script for the PDF ingestion pipeline in `ai_module.core.ingest`.

This script provides a command-line interface to run the `load_pdf` function
on a specified PDF file and prints the structured output.

Usage:
    python test_ingest.py <path_to_your_pdf_file.pdf>

Example:
    python test_ingest.py data/sample-presentation.pdf
"""
import sys
import json
import argparse
import os

# Ensure the ai_module is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_module.core.ingest import load_pdf

def main():
    parser = argparse.ArgumentParser(description="Test the PDF ingestion pipeline from ingest.py.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest.")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: The file '{args.pdf_path}' was not found.")
        sys.exit(1)

    print(f"Ingesting PDF: {args.pdf_path}")
    results = load_pdf("args.pdf_path")
    print("\n--- Ingestion Results ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSuccessfully processed {len(results)} pages from '{os.path.basename(args.pdf_path)}'.")

if __name__ == "__main__":
    main()
