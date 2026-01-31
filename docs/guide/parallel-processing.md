# Parallel Processing

Extract multiple pages concurrently for faster processing.

## Basic Parallel Extraction

```python
from xtra import create_extractor, ExtractorType

with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract(max_workers=2)
    print(f"Extracted {len(result.document.pages)} pages")
```

## Executor Types

xtra supports two executor types for parallel processing:

### Thread Executor (Default)

Best for most OCR use cases. Threads share the model cache and have low overhead.

```python
from xtra import create_extractor, ExtractorType, ExecutorType

with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract(max_workers=2, executor=ExecutorType.THREAD)
    print(f"Thread executor: {len(result.document.pages)} pages")
```

### Process Executor

Best for CPU-bound pure Python workloads. Models are duplicated per worker, resulting in higher memory usage.

```python
from xtra import create_extractor, ExtractorType, ExecutorType

with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract(max_workers=2, executor=ExecutorType.PROCESS)
    print(f"Process executor: {len(result.document.pages)} pages")
```

## Comparison

| Executor | Best For | Notes |
|----------|----------|-------|
| `THREAD` (default) | Most OCR use cases | Shared model cache, low overhead, C libraries release GIL |
| `PROCESS` | CPU-bound pure Python | Models duplicated per worker, higher memory usage |

## Extracting Specific Pages in Parallel

```python
from xtra import create_extractor, ExtractorType

with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
    result = extractor.extract(pages=[0, 1], max_workers=2)
    print(f"Extracted pages: {[p.page for p in result.document.pages]}")
```

## LLM Parallel Extraction

LLM extraction also supports parallel processing:

<!-- skip: next -->
```python
from xtra.llm import extract_structured

result = extract_structured(
    "document.pdf",
    model="openai/gpt-4o",
    max_workers=4,
)
# result.data is a list of per-page results
```
