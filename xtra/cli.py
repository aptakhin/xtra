#!/usr/bin/env python
"""CLI script to extract text from PDF and image files."""

from __future__ import annotations

# Suppress FutureWarning from instructor's internal google.generativeai import
# Must be before any imports that might trigger it
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="instructor")

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xtra.extractors.base import ExecutorType
from xtra.extractors.factory import CHARACTER_MERGER_CHOICES, create_extractor
from xtra.models import CoordinateUnit, ExtractorType


def _build_credentials(args: argparse.Namespace) -> dict[str, str] | None:
    """Build credentials dict from CLI arguments."""
    credentials: dict[str, str] = {}

    if args.azure_endpoint:
        credentials["XTRA_AZURE_DI_ENDPOINT"] = args.azure_endpoint
    if args.azure_key:
        credentials["XTRA_AZURE_DI_KEY"] = args.azure_key
    if args.google_processor_name:
        credentials["XTRA_GOOGLE_DOCAI_PROCESSOR_NAME"] = args.google_processor_name
    if args.google_credentials_path:
        credentials["XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH"] = args.google_credentials_path

    return credentials if credentials else None


def _parse_headers(header_list: list[str] | None) -> dict[str, str] | None:
    """Parse header arguments into a dict."""
    if not header_list:
        return None
    headers = {}
    for header in header_list:
        if "=" not in header:
            print(f"Warning: Invalid header format '{header}', expected KEY=VALUE", file=sys.stderr)
            continue
        key, value = header.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers if headers else None


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
            character_merger=args.character_merger,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _print_llm_result(data: Any, as_json: bool) -> None:
    """Print LLM extraction result."""
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    # Simple key-value output for dicts, otherwise just print
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            elif isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
    else:
        print(data)


def _print_table(table: Any) -> None:
    """Print a table in ASCII format."""
    if not table.cells:
        print("(empty table)")
        return

    # Build grid from cells
    grid: dict[tuple[int, int], str] = {}
    for cell in table.cells:
        grid[(cell.row, cell.col)] = cell.text

    # Calculate column widths
    col_widths: dict[int, int] = {}
    for col in range(table.col_count):
        col_widths[col] = max(len(grid.get((row, col), "")) for row in range(table.row_count))
        col_widths[col] = max(col_widths[col], 3)  # Minimum width

    # Print rows
    for row in range(table.row_count):
        cells = [grid.get((row, col), "").ljust(col_widths[col]) for col in range(table.col_count)]
        print("| " + " | ".join(cells) + " |")
        if row == 0:
            # Print header separator
            sep = ["-" * col_widths[col] for col in range(table.col_count)]
            print("|-" + "-|-".join(sep) + "-|")


def _attach_tables_to_pages(result: Any, tables: list[Any]) -> None:
    """Attach extracted tables to their respective pages."""
    tables_by_page: dict[int, list[Any]] = {}
    for table in tables:
        tables_by_page.setdefault(table.page, []).append(table)

    for page in result.document.pages:
        page.tables = tables_by_page.get(page.page, [])


def _extract_and_attach_tables(
    extractor: Any,
    result: Any,
    pages: Sequence[int] | None,
    table_mode: str | None,
) -> None:
    """Extract tables and attach them to document pages."""
    table_options: dict[str, Any] = {}
    if table_mode == "lattice":
        table_options["lattice"] = True
    elif table_mode == "stream":
        table_options["stream"] = True

    tables = extractor.extract_tables(pages=pages, table_options=table_options)
    _attach_tables_to_pages(result, tables)


def _extract_paddle_tables(
    extractor: Any,
    result: Any,
    pages: Sequence[int] | None,
) -> None:
    """Extract tables from PaddleOCR using PPStructure."""
    try:
        from paddleocr import PPStructure  # type: ignore[attr-defined] # noqa: F401
    except ImportError:
        print(
            "Warning: PPStructure not available. Install with: pip install 'paddleocr>=2.6'",
            file=sys.stderr,
        )
        return

    pages_list = list(pages) if pages else None
    tables = extractor.extract_tables(pages=pages_list)
    _attach_tables_to_pages(result, tables)


def _run_llm_extraction(args: argparse.Namespace, pages: Sequence[int] | None) -> None:
    """Run LLM-based extraction."""
    try:
        from xtra.llm.factory import extract_structured
    except ImportError as e:
        print(f"Error: LLM dependencies not installed: {e}", file=sys.stderr)
        sys.exit(1)

    credentials = _build_credentials(args)
    headers = _parse_headers(args.headers)
    pages_list = list(pages) if pages else None

    try:
        result = extract_structured(
            path=args.input,
            model=args.llm,
            prompt=args.prompt,
            pages=pages_list,
            dpi=args.dpi,
            credentials=credentials,
            base_url=args.base_url,
            headers=headers,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    _print_llm_result(result.data, args.json)


def _setup_parser() -> argparse.ArgumentParser:
    """Set up argument parser with all CLI options."""
    parser = argparse.ArgumentParser(description="Extract text from PDF/image files")
    parser.add_argument("input", type=Path, help="Input file path")
    parser.add_argument(
        "--extractor",
        type=str,
        choices=[e.value for e in ExtractorType],
        default=None,
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
        "--character-merger",
        type=str,
        choices=list(CHARACTER_MERGER_CHOICES.keys()),
        default=None,
        help="Character merger for PDF extractor: basic-line (default), keep-char",
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Extract tables (PDF requires tabula-py; Azure/Google extract automatically)",
    )
    parser.add_argument(
        "--table-mode",
        type=str,
        choices=["lattice", "stream"],
        default=None,
        help="Table extraction mode: lattice (bordered tables), stream (borderless)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for page extraction (default: 1, sequential)",
    )
    parser.add_argument(
        "--executor",
        type=str,
        choices=["thread", "process"],
        default="thread",
        help="Executor type for parallel extraction: thread (default), process",
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
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM model for extraction (e.g., gpt-4o, claude-3-5-sonnet, azure-openai/gpt-4o)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for LLM extraction (default: extract all key-value pairs)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom API base URL for OpenAI-compatible LLMs (vLLM, Ollama, etc.)",
    )
    parser.add_argument(
        "--header",
        type=str,
        action="append",
        dest="headers",
        metavar="KEY=VALUE",
        help="Custom HTTP header (can be repeated). Example: --header 'Authorization=Bearer token'",
    )
    return parser


def main() -> None:
    parser = _setup_parser()
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate: either --extractor or --llm must be provided
    if not args.extractor and not args.llm:
        print("Error: Either --extractor or --llm must be specified", file=sys.stderr)
        sys.exit(1)

    pages: Sequence[int] | None = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]

    # LLM extraction path
    if args.llm:
        _run_llm_extraction(args, pages)
        return

    # Regular OCR extraction path
    languages = [lang.strip() for lang in args.lang.split(",")]
    extractor = _create_extractor(args, languages)
    executor_type = ExecutorType(args.executor)

    with extractor:
        result = extractor.extract(pages=pages, max_workers=args.workers, executor=executor_type)

        # Extract tables if requested
        if args.tables:
            if args.extractor == "pdf":
                _extract_and_attach_tables(extractor, result, pages, args.table_mode)
            elif args.extractor in ("azure-di", "google-docai"):
                # Tables are extracted automatically by these extractors
                pass
            elif args.extractor == "paddle":
                # PaddleOCR uses PPStructure for table extraction
                _extract_paddle_tables(extractor, result, pages)
            else:
                print(
                    f"Warning: --tables is not supported for {args.extractor} extractor",
                    file=sys.stderr,
                )

    doc = result.document
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
            # Print tables if present
            for i, table in enumerate(page.tables):
                print(f"\n--- Table {i + 1} ({table.row_count}x{table.col_count}) ---")
                _print_table(table)
            print()


if __name__ == "__main__":
    main()
