#!/usr/bin/env python
"""CLI script to extract text from PDF and image files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .extractors import (
    AzureDocumentIntelligenceExtractor,
    EasyOcrExtractor,
    PdfExtractor,
    PdfToImageEasyOcrExtractor,
)
from .models import SourceType


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDF/image files")
    parser.add_argument("input", type=Path, help="Input file path")
    parser.add_argument(
        "--extractor",
        type=str,
        choices=[e.value for e in SourceType],
        required=True,
        help="Extractor type: pdf, easyocr, pdf-easyocr, tesseract, pdf-tesseract, paddle, "
        "pdf-paddle, azure-di, google-docai",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="OCR languages, comma-separated (default: en)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page numbers to extract, comma-separated (default: all). Example: 0,1,2",
    )
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help="Azure Document Intelligence endpoint URL (required for azure-di)",
    )
    parser.add_argument(
        "--azure-key",
        type=str,
        default=None,
        help="Azure Document Intelligence API key (required for azure-di)",
    )
    parser.add_argument(
        "--azure-model",
        type=str,
        default="prebuilt-read",
        help="Azure Document Intelligence model ID (default: prebuilt-read)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    languages = [lang.strip() for lang in args.lang.split(",")]
    pages: Optional[Sequence[int]] = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]

    extractor_type = SourceType(args.extractor)

    if extractor_type == SourceType.PDF:
        extractor = PdfExtractor(args.input)
    elif extractor_type == SourceType.EASYOCR:
        extractor = EasyOcrExtractor(args.input, languages=languages)
    elif extractor_type == SourceType.PDF_EASYOCR:
        extractor = PdfToImageEasyOcrExtractor(args.input, languages=languages)
    elif extractor_type == SourceType.AZURE_DI:
        if not args.azure_endpoint or not args.azure_key:
            print(
                "Error: --azure-endpoint and --azure-key are required for azure-di",
                file=sys.stderr,
            )
            sys.exit(1)
        extractor = AzureDocumentIntelligenceExtractor(
            args.input,
            endpoint=args.azure_endpoint,
            key=args.azure_key,
            model_id=args.azure_model,
        )
    else:
        print(f"Error: Unknown extractor type: {args.extractor}", file=sys.stderr)
        sys.exit(1)

    with extractor:
        doc = extractor.extract(pages=pages)

    if args.json:
        # pydantic v2 uses model_dump_json, v1 uses json
        if hasattr(doc, "model_dump_json"):
            print(doc.model_dump_json(indent=2))
        else:
            print(doc.json(indent=2))
    else:
        for page in doc.pages:
            print(f"=== Page {page.page + 1} ===")
            for text in page.texts:
                bbox = text.bbox
                conf = f" ({text.confidence:.2f})" if text.confidence else ""
                print(
                    f"[{bbox.x0:.1f},{bbox.y0:.1f},{bbox.x1:.1f},{bbox.y1:.1f}]{conf} {text.text}"
                )
            print()


if __name__ == "__main__":
    main()
