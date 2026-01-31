# Installation

## Basic Installation

Install xtra using uv:

```bash
uv sync
```

## Optional Dependencies

xtra uses optional dependencies to keep the base installation lightweight. Install only what you need:

### PDF Extraction

```bash
uv sync --extra pdf
```

### Local OCR Engines

```bash
# EasyOCR
uv sync --extra easyocr

# Tesseract (requires system Tesseract installation)
uv sync --extra tesseract

# PaddleOCR
uv sync --extra paddle
```

### Cloud OCR Services

```bash
# Azure Document Intelligence
uv sync --extra azure

# Google Document AI
uv sync --extra google
```

### LLM Providers

```bash
# OpenAI
uv sync --extra llm-openai

# Anthropic
uv sync --extra llm-anthropic

# Google Gemini
uv sync --extra llm-google

# All LLM providers
uv sync --extra llm-all
```

### Everything

```bash
uv sync --extra all
```

## System Requirements

### Tesseract OCR

Tesseract requires system installation:

- **macOS**: `brew install tesseract`
- **Ubuntu**: `apt-get install tesseract-ocr`
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### PaddleOCR

PaddleOCR works out of the box but may require additional setup for GPU acceleration.
