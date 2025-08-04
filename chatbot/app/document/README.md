# Document Generation Module

This module provides document generation capabilities for the TinyLLM Chatbot, allowing users to export LLM responses into various document formats.

## Features

- **PDF Generation**: Creates formatted PDF documents using ReportLab
- **Word Documents**: Generates `.docx` files using python-docx
- **Excel Spreadsheets**: Creates `.xlsx` files using openpyxl
- **PowerPoint Presentations**: Generates `.pptx` files using python-pptx

## Architecture

### Files

- `document_generator.py`: Main document generation logic
- `__init__.py`: Package initialization and exports

### Key Classes

#### DocumentGenerator
Main class that handles document creation for all supported formats.

**Methods:**
- `generate_document()`: Main entry point for document generation
- `create_document_file()`: Routes to appropriate format-specific creator
- `create_pdf_document()`: PDF generation using ReportLab
- `create_word_document()`: Word document generation using python-docx
- `create_excel_document()`: Excel spreadsheet generation using openpyxl
- `create_powerpoint_document()`: PowerPoint presentation generation using python-pptx
- `is_supported_format()`: Check if format is supported
- `get_supported_formats()`: Get list of all supported formats

## Usage

### Basic Usage

```python
from app.document import generate_document_from_response

# Generate a document from an LLM response
file_path = await generate_document_from_response(
    session_id="abc123",
    llm_response="Your detailed analysis content here...",
    doc_request={
        "type": "pdf",
        "original_prompt": "Create a PDF report about renewable energy",
        "requested": True,
        "content_ready": True
    },
    model="gpt-4"
)
```

### Direct DocumentGenerator Usage

```python
from app.document import get_document_generator

doc_gen = get_document_generator()
file_path = await doc_gen.generate_document(
    session_id, llm_response, doc_request, model
)
```

## Integration with Chatbot

The document generation is integrated into the chatbot's intent detection system:

1. **Intent Detection**: When users request document generation (e.g., "Create a PDF of this analysis"), the intent router detects the request
2. **Content Extraction**: The system extracts the main content request while including conversation context
3. **Document Type Detection**: LLM determines the requested document format
4. **Content Processing**: LLM processes and responds to the content request normally
5. **Document Generation**: After LLM response completion, the document generator creates the requested format
6. **Download Link**: User receives a download link for the generated document

## Example User Interactions

```
User: "I need an analysis of renewable energy trends in 2024"
Assistant: [Provides detailed analysis]
User: "Create a PDF report of this analysis"
Assistant: [Document Generation Requested: PDF]
          [Generating PDF document...]
          [Document Generated Successfully]
```

## Supported Document Types

| Format | File Extension | Library Used |
|--------|---------------|--------------|
| PDF | `.pdf` | reportlab |
| Word | `.docx` | python-docx |
| Excel | `.xlsx` | openpyxl |
| PowerPoint | `.pptx` | python-pptx |

## File Structure

Generated documents are stored in:
```
chatbot/generated_docs/
├── document_abc12345_20250803_143022.pdf
├── document_def67890_20250803_143105.docx
└── ...
```

Filename format: `document_{session_id[:8]}_{timestamp}.{extension}`

## Dependencies

The module requires the following Python packages:
- `reportlab` - PDF generation
- `python-docx` - Word document generation  
- `openpyxl` - Excel spreadsheet generation
- `python-pptx` - PowerPoint presentation generation

These are automatically installed via the Dockerfile.

## Error Handling

The module includes comprehensive error handling:
- **Import Errors**: Gracefully handles missing dependencies
- **File Creation Errors**: Returns None if document creation fails
- **Permission Errors**: Logs errors for debugging
- **Content Parsing Errors**: Handles malformed content gracefully

## Future Enhancements

Potential improvements:
- **Custom Templates**: Support for document templates
- **Enhanced Formatting**: Better table and image handling
- **Batch Generation**: Multiple documents in one request
- **Custom Styling**: User-defined document styles
- **Document Merging**: Combine multiple responses into one document
