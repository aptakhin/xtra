# Configuration Reference

## PDF Extractor Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `Path \| str` | required | Path to the PDF file |
| `output_unit` | `CoordinateUnit` | `POINTS` | Output coordinate unit |
| `character_merger` | `CharacterMerger` | `BasicLineMerger()` | Text merging strategy |

### Extract Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pages` | `Sequence[int]` | `None` | Pages to extract (0-indexed). `None` = all pages |
| `max_workers` | `int` | `1` | Number of parallel workers |
| `executor` | `ExecutorType` | `THREAD` | Executor type for parallelism |
| `table_options` | `dict` | `None` | Tabula options for table extraction |

## Table Extraction Options (Tabula)

Pass these options via the `table_options` parameter to enable table extraction.

### Common Options

| Option | Type | Description |
|--------|------|-------------|
| `lattice` | `bool` | Use lattice mode for tables with visible cell borders |
| `stream` | `bool` | Use stream mode for tables without visible borders |
| `guess` | `bool` | Automatically guess table areas |
| `multiple_tables` | `bool` | Extract multiple tables per page |

### Area Options

| Option | Type | Description |
|--------|------|-------------|
| `area` | `tuple[float, float, float, float]` | Extract area: (top, left, bottom, right) in points |
| `columns` | `list[float]` | X-coordinates for column splitting |

### Example

```python skip: requires tabula
from xtra import PdfExtractor

with PdfExtractor("table.pdf") as extractor:
    # Lattice mode for bordered tables
    result = extractor.extract(table_options={"lattice": True})

    # Stream mode for borderless tables
    result = extractor.extract(table_options={"stream": True})

    # Extract from specific area
    result = extractor.extract(table_options={
        "area": (100, 50, 400, 500),  # top, left, bottom, right
        "lattice": True,
    })
```

## Environment Variables

### OCR Extractors

| Variable | Description | Default |
|----------|-------------|---------|
| `XTRA_AZURE_DI_ENDPOINT` | Azure Document Intelligence endpoint URL | - |
| `XTRA_AZURE_DI_KEY` | Azure Document Intelligence API key | - |
| `XTRA_AZURE_DI_MODEL` | Azure model ID | `prebuilt-read` |
| `XTRA_GOOGLE_DOCAI_PROCESSOR_NAME` | Google Document AI processor name | - |
| `XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH` | Path to Google service account JSON | - |

### LLM Providers

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version (default: `2024-02-15-preview`) |

## Coordinate Units

All extractors support the `output_unit` parameter to control the coordinate system of bounding boxes.

| Unit | Description | Use Case |
|------|-------------|----------|
| `POINTS` | 1/72 inch | PDF native, resolution-independent |
| `PIXELS` | Pixels at specified DPI | Image processing, display |
| `INCHES` | Imperial inches | Print layout |
| `NORMALIZED` | 0-1 range relative to page | ML models, relative positioning |

!!! note
    PDF extractor doesn't support `PIXELS` output because PDFs don't have inherent DPI.
    OCR extractors use the `dpi` parameter for pixel conversions.

## DPI Settings

OCR extractors accept a `dpi` parameter that affects:

1. **PDF to image conversion** - Higher DPI = larger images, better quality
2. **Coordinate conversion** - Used when converting between PIXELS and other units

Recommended values:

| Use Case | DPI |
|----------|-----|
| Fast preview | 72-100 |
| Standard OCR | 150-200 |
| High quality | 300+ |

## Parallel Processing

### max_workers

Number of parallel workers for page extraction. Default is 1 (sequential).

```python
result = extractor.extract(max_workers=4)
```

### executor

Type of executor for parallel processing:

- `ExecutorType.THREAD` (default) - Thread pool executor
- `ExecutorType.PROCESS` - Process pool executor

```python
from xtra import ExecutorType

result = extractor.extract(max_workers=4, executor=ExecutorType.PROCESS)
```
