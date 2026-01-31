# CLI Usage

xtra provides a command-line interface for document extraction.

## Basic Commands

### PDF Extraction

```bash
uv run python -m xtra.cli document.pdf --extractor pdf
```

### OCR Extraction

```bash
# EasyOCR (works for both images and PDFs)
uv run python -m xtra.cli image.png --extractor easyocr --lang en
uv run python -m xtra.cli scanned.pdf --extractor easyocr --lang en

# Tesseract
uv run python -m xtra.cli document.pdf --extractor tesseract --lang eng

# PaddleOCR
uv run python -m xtra.cli document.pdf --extractor paddle --lang en
```

### Cloud OCR

```bash
# Azure Document Intelligence
uv run python -m xtra.cli document.pdf --extractor azure-di \
    --azure-endpoint https://your-resource.cognitiveservices.azure.com \
    --azure-key your-api-key

# Google Document AI
uv run python -m xtra.cli document.pdf --extractor google-docai \
    --google-processor-name projects/your-project/locations/us/processors/123 \
    --google-credentials-path /path/to/credentials.json
```

## Parallel Processing

```bash
# Use 4 parallel workers
uv run python -m xtra.cli document.pdf --extractor pdf --workers 4

# Use process executor instead of threads
uv run python -m xtra.cli document.pdf --extractor pdf --workers 4 --executor process
```

## Output Formats

```bash
# JSON output
uv run python -m xtra.cli document.pdf --extractor pdf --json

# Specific pages
uv run python -m xtra.cli document.pdf --extractor pdf --pages 0,1
```

## LLM Extraction

```bash
# Free-form extraction
uv run python -m xtra.cli document.pdf --llm openai/gpt-4o

# With custom prompt
uv run python -m xtra.cli image.png --llm anthropic/claude-sonnet-4-20250514 \
    --llm-prompt "Extract all text from this image"

# With parallel workers
uv run python -m xtra.cli document.pdf --llm openai/gpt-4o --workers 4

# With OpenAI-compatible API
uv run python -m xtra.cli document.pdf --llm openai/llava \
    --llm-base-url http://localhost:11434/v1

# With custom headers
uv run python -m xtra.cli document.pdf --llm openai/gpt-4o \
    --llm-base-url https://your-proxy.com/v1 \
    --llm-header "X-Custom-Auth=your-token"

# JSON output
uv run python -m xtra.cli document.pdf --llm openai/gpt-4o --json
```

## Environment Variables

Instead of passing credentials via CLI, you can use environment variables:

```bash
# Azure
export XTRA_AZURE_DI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
export XTRA_AZURE_DI_KEY=your-api-key
uv run python -m xtra.cli document.pdf --extractor azure-di

# Google
export XTRA_GOOGLE_DOCAI_PROCESSOR_NAME=projects/your-project/locations/us/processors/123
export XTRA_GOOGLE_DOCAI_CREDENTIALS_PATH=/path/to/credentials.json
uv run python -m xtra.cli document.pdf --extractor google-docai

# LLM
export OPENAI_API_KEY=your-key
uv run python -m xtra.cli document.pdf --llm openai/gpt-4o
```
