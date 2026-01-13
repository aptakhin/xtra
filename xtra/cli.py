#!/usr/bin/env python
"""CLI script to extract text from PDF and image files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from xtra.extractors.factory import create_extractor
from xtra.models import CoordinateUnit, ExtractorType


def _build_credentials(args: argparse.Namespace) -> Optional[Dict[str, str]]:
    """Build credentials dict from CLI arguments."""
    credentials: Dict[str, str] = {}

    if args.azure_endpoint:
        credentials["XTRA_AZURE_DI_ENDPOINT"] = args.azure_endpoint
    if args.azure_key:
        credentials["XTRA_AZURE_DI_KEY"] = args.azure_key
    if args.google_processor_name:
        credentials["XTRA_GOOGLE_DOCAI_PROCESSOR_NAME"] = args.google_processor_name
    if args.google_credentials_path:
        credentials["XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH"] = args.google_credentials_path

    return credentials if credentials else None


def _create_extractor(args: argparse.Namespace, languages: list[str]) -> Any:
    """Create extractor using the unified factory."""
    extractor_type = ExtractorType(args.extractor)
    credentials = _build_credentials(args)

    # Parse output unit
    output_unit = CoordinateUnit(args.output_unit)

    try:
        return create_extractor(
            path=args.input,
            extractor_type=extractor_type,
            languages=languages,
            dpi=args.dpi,
            use_gpu=args.gpu,
            credentials=credentials,
            output_unit=output_unit,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from PDF/image files")
    parser.add_argument("input", type=Path, help="Input file path")
    parser.add_argument(
        "--extractor",
        type=str,
        choices=[e.value for e in ExtractorType],
        required=True,
        help="Extractor type: pdf, easyocr, tesseract, paddle, azure-di, google-docai",
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
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF-to-image conversion (default: 200)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration (EasyOCR, PaddleOCR)",
    )
    parser.add_argument(
        "--output-unit",
        type=str,
        choices=[u.value for u in CoordinateUnit],
        default="points",
        help="Coordinate unit for output: points (default), pixels, inches, normalized",
    )
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help="Azure endpoint URL (or XTRA_AZURE_DI_ENDPOINT env var)",
    )
    parser.add_argument(
        "--azure-key",
        type=str,
        default=None,
        help="Azure API key (or XTRA_AZURE_DI_KEY env var)",
    )
    parser.add_argument(
        "--google-processor-name",
        type=str,
        default=None,
        help="Google processor name (or XTRA_GOOGLE_DOCAI_PROCESSOR_NAME env var)",
    )
    parser.add_argument(
        "--google-credentials-path",
        type=str,
        default=None,
        help="Google service account JSON path (or XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    languages = [lang.strip() for lang in args.lang.split(",")]
    pages: Optional[Sequence[int]] = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]

    extractor = _create_extractor(args, languages)

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
