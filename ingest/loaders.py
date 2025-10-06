"""
Document loaders using Docling with HybridChunker.

Provides official langchain-docling integration for loading documents
with structure-aware chunking, OCR support, and rich metadata preservation.
Replaces manual markdown export + generic chunking approach (v1.0-1.2).
"""

import logging
from pathlib import Path
from typing import List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from langchain_core.documents import Document as LangChainDocument
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from settings import settings

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Document loader using langchain-docling with HybridChunker.
    
    Uses Docling's official LangChain integration for:
    - Structure-aware chunking (respects document hierarchy)
    - OCR support for scanned PDFs/images
    - Accurate table extraction (TableFormerMode.ACCURATE)
    - Rich metadata preservation (headings, page numbers, bounding boxes)
    - Tokenization-aware chunking (aligns with embedding models)
    
    This replaces the v1.0-1.2 approach of:
    - export_to_markdown() → loses structure
    - RecursiveCharacterTextSplitter/SemanticChunker → generic, structure-agnostic
    
    Attributes:
        converter: Configured Docling DocumentConverter with OCR and table settings
        chunker: HybridChunker for hierarchical + tokenization-aware chunking
    """
    
    def __init__(self):
        """Initialize DocumentLoader with optimized Docling configuration."""
        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = settings.docling_ocr
        pipeline_options.do_table_structure = True
        
        # Table extraction mode
        if settings.docling_table_mode == "accurate":
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        else:
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        
        # Create converter with PDF-specific settings
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        # HybridChunker: hierarchical + tokenization-aware
        self.chunker = HybridChunker()
        
        logger.info(
            f"Docling initialized: OCR={settings.docling_ocr}, "
            f"TableMode={settings.docling_table_mode}"
        )
    
    def load_from_directory(self, directory: Path) -> List[LangChainDocument]:
        """
        Load all supported documents from a directory.
        
        Uses DoclingLoader with HybridChunker for structure-aware chunking.
        Supports: PDF, DOCX, PPTX, XLSX, Markdown, HTML, and images.
        
        Returns LangChain Documents with rich metadata including:
        - Headings (section context)
        - Page numbers (grounding)
        - Bounding boxes (precise citations)
        - Document hierarchy preserved
        
        **Image Support (PNG, JPG, JPEG, TIFF, BMP):**
        Images are automatically processed with OCR when ARQ_DOCLING_OCR=true.
        Docling's default pipeline applies OCR to all image-based content without
        needing explicit ImageFormatOption configuration. For advanced image-specific
        settings (resolution, preprocessing, etc.), ImageFormatOption can be added
        as a future enhancement.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of LangChain Document objects with rich metadata
            
        Raises:
            FileNotFoundError: If directory doesn't exist
            
        Example:
            >>> loader = DocumentLoader()
            >>> docs = loader.load_from_directory(Path("./data"))
            >>> print(f"Loaded {len(docs)} chunks with structure metadata")
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        logger.info(f"Scanning directory for documents: {directory}")
        
        # Supported extensions per Docling documentation
        supported_extensions = {
            ".pdf", ".docx", ".pptx", ".xlsx",
            ".md", ".markdown", ".html", ".htm",
            ".png", ".jpg", ".jpeg", ".tiff", ".bmp"
        }
        
        all_documents = []
        file_count = 0
        success_count = 0
        
        # Process each file
        for file_path in directory.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    logger.info(f"Processing: {file_path.name}")
                    file_count += 1
                    
                    # Use DoclingLoader with configured converter and chunker
                    loader = DoclingLoader(
                        file_path=str(file_path),
                        converter=self.converter,
                        export_type=ExportType.DOC_CHUNKS,  # Use HybridChunker!
                        chunker=self.chunker
                    )
                    
                    # Load and chunk in one step
                    documents = loader.load()
                    
                    # Add relative path to metadata
                    for doc in documents:
                        doc.metadata["relative_path"] = str(
                            file_path.relative_to(directory)
                        )
                    
                    all_documents.extend(documents)
                    success_count += 1
                    
                    logger.debug(
                        f"✓ Loaded {file_path.name}: {len(documents)} chunks"
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
                    continue
        
        logger.info(
            f"Successfully loaded {success_count}/{file_count} documents "
            f"({len(all_documents)} total chunks)"
        )
        return all_documents
    
    def load_single_file(self, file_path: Path) -> List[LangChainDocument]:
        """
        Load a single document file with structure-aware chunking.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document chunks
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading single file: {file_path}")
        
        try:
            loader = DoclingLoader(
                file_path=str(file_path),
                converter=self.converter,
                export_type=ExportType.DOC_CHUNKS,
                chunker=self.chunker
            )
            
            documents = loader.load()
            
            logger.info(
                f"✓ Loaded {file_path.name}: {len(documents)} chunks"
            )
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise ValueError(f"Unsupported or corrupted file: {file_path}") from e
