# Async API

xtra provides native async/await support for integration with async applications.

## Basic Async Extraction

```python
import asyncio
from xtra import create_extractor, ExtractorType

async def extract_document():
    with create_extractor("document.pdf", ExtractorType.PDF) as extractor:
        result = await extractor.extract_async(max_workers=2)
        return result.document

doc = asyncio.run(extract_document())
print(f"Extracted {len(doc.pages)} pages asynchronously")
```

## Async LLM Extraction

<!-- skip: next -->
```python
import asyncio
from xtra.llm import extract_structured_async

async def extract():
    result = await extract_structured_async(
        "document.pdf",
        model="openai/gpt-4o",
        max_workers=4,
    )
    return result.data

data = asyncio.run(extract())
```

## Using with FastAPI

<!-- skip: next -->
```python
from fastapi import FastAPI, UploadFile
from xtra import create_extractor, ExtractorType

app = FastAPI()

@app.post("/extract")
async def extract_document(file: UploadFile):
    # Save uploaded file temporarily
    content = await file.read()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    # Extract asynchronously
    with create_extractor(temp_path, ExtractorType.PDF) as extractor:
        result = await extractor.extract_async()
        return {"pages": len(result.document.pages)}
```

## Concurrent Document Processing

Process multiple documents concurrently:

```python
import asyncio
from xtra import create_extractor, ExtractorType

async def extract_one(path: str):
    with create_extractor(path, ExtractorType.PDF) as extractor:
        return await extractor.extract_async()

async def extract_many(paths: list[str]):
    tasks = [extract_one(p) for p in paths]
    return await asyncio.gather(*tasks)

# Using the same file twice for demo
paths = ["document.pdf", "document.pdf"]
results = asyncio.run(extract_many(paths))
print(f"Extracted {len(results)} documents concurrently")
```
