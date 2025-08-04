"""
Document Generation Module for TinyLLM Chatbot

This module handles document generation functionality including PDF, Word, Excel, 
and PowerPoint document creation based on LLM responses.

Author: Jason A. Cox
3 Aug 2025
github.com/jasonacox/TinyLLM
"""

import os
import datetime
from typing import Optional, Dict, Any

from app.core.config import log, debug
from app.core.llm import ask_llm

# Create directory for generated documents
# Get the chatbot directory (3 levels up from this file)
_current_file = os.path.abspath(__file__)
_chatbot_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_file)))
GENERATED_DOCS_DIR = os.path.join(_chatbot_dir, "generated_docs")
os.makedirs(GENERATED_DOCS_DIR, exist_ok=True)


class DocumentGenerator:
    """Handles document generation for various formats."""
    
    def __init__(self):
        self.supported_formats = ["pdf", "word", "doc", "docx", "excel", "spreadsheet", "xlsx", "powerpoint", "ppt", "pptx"]
    
    async def generate_document(self, session_id: str, llm_response: str, doc_request: Dict[str, Any], model: str) -> Optional[str]:
        """
        Generate document based on LLM response and user request.
        
        Args:
            session_id: Session identifier
            llm_response: The LLM's response content
            doc_request: Document request configuration
            model: LLM model to use for structuring
            
        Returns:
            str: Generated file path or None if failed
        """
        try:
            debug(f"Starting document generation for session {session_id[:8]}")
            debug(f"Document type: {doc_request['type']}")
            debug(f"Original prompt: {doc_request.get('original_prompt', 'Not provided')}")
            debug(f"LLM response length: {len(llm_response)} characters")
            debug(f"Model: {model}")
            
            log(f"Generating {doc_request['type'].upper()} document for session {session_id[:8]}")
            
            # Step 1: Ask LLM to structure the content for document
            debug("Step 1: Structuring content with LLM")
            structure_prompt = f"""
            The user originally requested: {doc_request["original_prompt"]}
            
            Here is the response content: {llm_response}
            
            Please structure this content appropriately for a {doc_request["type"]} document.
            For the format, provide:
            - A clear title
            - Organized sections with headers
            - Any data that should be formatted as tables
            - Proper formatting suggestions
            
            Return the structured content in a clear, organized format.
            """
            
            debug(f"Sending structure prompt to LLM (length: {len(structure_prompt)} chars)")
            structured_content = await ask_llm(structure_prompt, model=model)
            debug(f"Received structured content from LLM (length: {len(structured_content)} chars)")
            
            # Step 2: Generate the actual document
            debug("Step 2: Creating document file")
            file_path = await self.create_document_file(session_id, structured_content, doc_request["type"])
            
            if file_path:
                debug(f"Document generation successful: {file_path}")
                log(f"Document generated successfully: {file_path}")
            else:
                debug("Document generation failed: file_path is None")
                log(f"Document generation failed for session {session_id[:8]}")
            
            return file_path
            
        except Exception as e:
            debug(f"Exception in generate_document: {type(e).__name__}: {e}")
            log(f"Document generation error: {e}")
            return None
    
    async def create_document_file(self, session_id: str, content: str, doc_type: str) -> Optional[str]:
        """Create the actual document file."""
        try:
            debug(f"Creating document file for session {session_id[:8]}")
            debug(f"Document type: {doc_type}")
            debug(f"Content length: {len(content)} characters")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"document_{session_id[:8]}_{timestamp}"
            debug(f"Generated filename: {filename}")
            
            # Check if output directory exists
            debug(f"Checking if output directory exists: {GENERATED_DOCS_DIR}")
            if not os.path.exists(GENERATED_DOCS_DIR):
                debug(f"Creating output directory: {GENERATED_DOCS_DIR}")
                os.makedirs(GENERATED_DOCS_DIR, exist_ok=True)
            
            if doc_type in ["pdf"]:
                debug("Creating PDF document")
                file_path = await self.create_pdf_document(content, filename)
            elif doc_type in ["word", "doc", "docx"]:
                debug("Creating Word document")
                file_path = await self.create_word_document(content, filename)
            elif doc_type in ["excel", "spreadsheet", "xlsx"]:
                debug("Creating Excel document")
                file_path = await self.create_excel_document(content, filename)
            elif doc_type in ["powerpoint", "ppt", "pptx"]:
                debug("Creating PowerPoint document")
                file_path = await self.create_powerpoint_document(content, filename)
            else:
                debug(f"Unknown document type '{doc_type}', defaulting to PDF")
                file_path = await self.create_pdf_document(content, filename)
            
            if file_path:
                debug(f"Document file created successfully: {file_path}")
            else:
                debug("Document file creation failed")
            
            return file_path
            
        except Exception as e:
            debug(f"Exception in create_document_file: {type(e).__name__}: {e}")
            log(f"Document creation error: {e}")
            return None
    
    async def create_pdf_document(self, content: str, filename: str) -> Optional[str]:
        """Create PDF using reportlab."""
        try:
            debug(f"Starting PDF creation for filename: {filename}")
            debug(f"Content length: {len(content)} characters")
            
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            debug("Successfully imported reportlab modules")
            
            file_path = os.path.join(GENERATED_DOCS_DIR, f"{filename}.pdf")
            debug(f"Target file path: {file_path}")
            
            # Create PDF document
            debug("Creating SimpleDocTemplate")
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Parse content into paragraphs
            lines = content.split('\n')
            debug(f"Parsed content into {len(lines)} lines")
            
            processed_lines = 0
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.2*inch))
                    continue
                
                # Determine style based on content
                if line.startswith('#') or (len(line) < 50 and line.isupper()):
                    # Heading
                    story.append(Paragraph(line.replace('#', '').strip(), styles['Heading1']))
                    debug(f"Added heading: {line[:50]}...")
                elif line.startswith('##'):
                    # Subheading
                    story.append(Paragraph(line.replace('##', '').strip(), styles['Heading2']))
                    debug(f"Added subheading: {line[:50]}...")
                else:
                    # Regular paragraph
                    story.append(Paragraph(line, styles['Normal']))
                
                story.append(Spacer(1, 0.1*inch))
                processed_lines += 1
            
            debug(f"Processed {processed_lines} content lines into {len(story)} story elements")
            debug("Building PDF document")
            doc.build(story)
            debug(f"PDF document built successfully: {filename}.pdf")
            
            return f"{filename}.pdf"
            
        except Exception as e:
            debug(f"Exception in create_pdf_document: {type(e).__name__}: {e}")
            log(f"PDF creation error: {e}")
            return None
    
    async def create_word_document(self, content: str, filename: str) -> Optional[str]:
        """Create Word doc using python-docx."""
        try:
            debug(f"Starting Word document creation for filename: {filename}")
            debug(f"Content length: {len(content)} characters")
            
            from docx import Document
            debug("Successfully imported python-docx")
            
            file_path = os.path.join(GENERATED_DOCS_DIR, f"{filename}.docx")
            debug(f"Target file path: {file_path}")
            
            # Create Word document
            debug("Creating new Word document")
            doc = Document()
            
            # Parse content into paragraphs
            lines = content.split('\n')
            debug(f"Parsed content into {len(lines)} lines")
            
            processed_lines = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Determine style based on content
                if line.startswith('#') or (len(line) < 50 and line.isupper()):
                    # Heading
                    doc.add_heading(line.replace('#', '').strip(), level=1)
                    debug(f"Added heading level 1: {line[:50]}...")
                elif line.startswith('##'):
                    # Subheading
                    doc.add_heading(line.replace('##', '').strip(), level=2)
                    debug(f"Added heading level 2: {line[:50]}...")
                else:
                    # Regular paragraph
                    doc.add_paragraph(line)
                
                processed_lines += 1
            
            debug(f"Processed {processed_lines} content lines")
            debug("Saving Word document")
            doc.save(file_path)
            debug(f"Word document saved successfully: {filename}.docx")
            
            return f"{filename}.docx"
            
        except Exception as e:
            debug(f"Exception in create_word_document: {type(e).__name__}: {e}")
            log(f"Word document creation error: {e}")
            return None
    
    async def create_excel_document(self, content: str, filename: str) -> Optional[str]:
        """Create Excel using openpyxl.""" 
        try:
            debug(f"Starting Excel document creation for filename: {filename}")
            debug(f"Content length: {len(content)} characters")
            
            from openpyxl import Workbook
            from openpyxl.styles import Font
            debug("Successfully imported openpyxl")
            
            file_path = os.path.join(GENERATED_DOCS_DIR, f"{filename}.xlsx")
            debug(f"Target file path: {file_path}")
            
            # Create Excel workbook
            debug("Creating new Excel workbook")
            wb = Workbook()
            ws = wb.active
            ws.title = "Generated Content"
            debug("Created active worksheet: Generated Content")
            
            # Parse content and add to spreadsheet
            lines = content.split('\n')
            debug(f"Parsed content into {len(lines)} lines")
            row = 1
            
            processed_lines = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this looks like tabular data
                if '|' in line and not line.startswith('#'):
                    # Table row
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    debug(f"Processing table row with {len(cells)} cells at row {row}")
                    for col, cell in enumerate(cells, 1):
                        ws.cell(row=row, column=col, value=cell)
                else:
                    # Regular content in first column
                    cell = ws.cell(row=row, column=1, value=line)
                    if line.startswith('#') or (len(line) < 50 and line.isupper()):
                        # Make headings bold
                        cell.font = Font(bold=True, size=14)
                        debug(f"Added heading at row {row}: {line[:50]}...")
                    else:
                        debug(f"Added content at row {row}: {line[:50]}...")
                
                row += 1
                processed_lines += 1
            
            debug(f"Processed {processed_lines} content lines into {row-1} Excel rows")
            debug("Saving Excel document")
            wb.save(file_path)
            debug(f"Excel document saved successfully: {filename}.xlsx")
            
            return f"{filename}.xlsx"
            
        except Exception as e:
            debug(f"Exception in create_excel_document: {type(e).__name__}: {e}")
            log(f"Excel creation error: {e}")
            return None
    
    async def create_powerpoint_document(self, content: str, filename: str) -> Optional[str]:
        """Create PowerPoint using python-pptx."""
        try:
            debug(f"Starting PowerPoint document creation for filename: {filename}")
            debug(f"Content length: {len(content)} characters")
            
            from pptx import Presentation
            debug("Successfully imported python-pptx")
            
            file_path = os.path.join(GENERATED_DOCS_DIR, f"{filename}.pptx")
            debug(f"Target file path: {file_path}")
            
            # Create PowerPoint presentation
            debug("Creating new PowerPoint presentation")
            prs = Presentation()
            
            # Parse content into slides
            lines = content.split('\n')
            debug(f"Parsed content into {len(lines)} lines")
            
            current_slide = None
            content_text = None
            slide_count = 0
            processed_lines = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Determine if this should be a new slide (headings)
                if line.startswith('#') or (len(line) < 50 and line.isupper() and current_slide is None):
                    # Create new slide
                    debug(f"Creating new slide with title: {line[:50]}...")
                    slide_layout = prs.slide_layouts[1]  # Title and Content layout
                    current_slide = prs.slides.add_slide(slide_layout)
                    title = current_slide.shapes.title
                    title.text = line.replace('#', '').strip()
                    content_placeholder = current_slide.placeholders[1]
                    content_text = content_placeholder.text_frame
                    slide_count += 1
                elif current_slide is not None and content_text is not None:
                    # Add content to current slide
                    p = content_text.add_paragraph()
                    p.text = line
                    debug(f"Added content to slide: {line[:30]}...")
                
                processed_lines += 1
            
            # If no slides were created, create a single slide with all content
            if len(prs.slides) == 0:
                debug("No slides created from content, creating default slide")
                slide_layout = prs.slide_layouts[1]
                slide = prs.slides.add_slide(slide_layout)
                title = slide.shapes.title
                title.text = "Generated Content"
                content_placeholder = slide.placeholders[1]
                content_placeholder.text = content
                slide_count = 1
            
            debug(f"Created {slide_count} slides from {processed_lines} content lines")
            debug("Saving PowerPoint presentation")
            prs.save(file_path)
            debug(f"PowerPoint presentation saved successfully: {filename}.pptx")
            
            return f"{filename}.pptx"
            
        except Exception as e:
            debug(f"Exception in create_powerpoint_document: {type(e).__name__}: {e}")
            log(f"PowerPoint creation error: {e}")
            return None
    
    def is_supported_format(self, doc_type: str) -> bool:
        """Check if document type is supported."""
        return doc_type.lower() in self.supported_formats
    
    def get_supported_formats(self) -> list:
        """Get list of supported document formats."""
        return self.supported_formats.copy()


# Global document generator instance
document_generator = DocumentGenerator()


async def generate_document_from_response(session_id: str, llm_response: str, doc_request: Dict[str, Any], model: str) -> Optional[str]:
    """
    Convenience function to generate document from LLM response.
    
    Args:
        session_id: Session identifier
        llm_response: The LLM's response content
        doc_request: Document request configuration
        model: LLM model to use
        
    Returns:
        str: Generated file path or None if failed
    """
    return await document_generator.generate_document(session_id, llm_response, doc_request, model)


def get_document_generator() -> DocumentGenerator:
    """Get the global document generator instance."""
    return document_generator
