# Models Reference

## Core Models

### Document

The top-level container for extracted content.

::: xtra.base.Document

### Page

Represents a single page with text blocks and tables.

::: xtra.base.Page

### TextBlock

A text element with bounding box and confidence.

::: xtra.base.TextBlock

### BBox

Bounding box coordinates.

::: xtra.base.BBox

### Table

Extracted table with rows and cells.

::: xtra.base.Table

### TableCell

Individual cell within a table.

::: xtra.base.TableCell

## Result Models

### ExtractionResult

Result of document extraction.

::: xtra.base.ExtractionResult

### PageExtractionResult

Result of single page extraction.

::: xtra.base.PageExtractionResult

## Metadata

### ExtractorMetadata

Metadata about the extractor used.

::: xtra.base.ExtractorMetadata
