# services/file_handler_service.py
import logging
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class FileHandlerService:
    """
    Service for reading file content with support for various file types.
    """

    def __init__(self):
        """Initialize the file handler service."""
        self.supported_text_extensions = {
            '.txt', '.md', '.markdown', '.rst',
            '.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
            '.toml', '.ini', '.cfg', '.conf', '.env',
            '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.swift', '.php', '.rb'
        }
        self.supported_doc_extensions = {'.pdf', '.docx'}
        logger.info("FileHandlerService initialized")

    def read_file_content(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Read content from a file.

        Args:
            file_path: Path to the file to read

        Returns:
            Tuple of (content, file_type, error_message)
            - content: The file content as string, or None if error/binary
            - file_type: "text", "binary", or "error"
            - error_message: Error description if file_type is "error", else None
        """
        if not os.path.exists(file_path):
            return None, "error", f"File not found: {file_path}"

        if not os.path.isfile(file_path):
            return None, "error", f"Path is not a file: {file_path}"

        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return "", "text", None

            # Check file size (50MB limit as per constants)
            max_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_size:
                return None, "error", f"File too large: {file_size / (1024 * 1024):.1f}MB > 50MB"

        except OSError as e:
            return None, "error", f"Cannot access file: {e}"

        file_ext = os.path.splitext(file_path)[1].lower()

        # Handle different file types
        if file_ext in self.supported_text_extensions:
            return self._read_text_file(file_path)
        elif file_ext in self.supported_doc_extensions:
            return self._read_document_file(file_path, file_ext)
        else:
            # Try to read as text, but check if it's binary
            return self._read_unknown_file(file_path)

    def _read_text_file(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Read a text file with encoding detection."""
        # Try common encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                return content, "text", None
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path} with {encoding}: {e}")
                continue

        return None, "error", f"Could not decode file with any supported encoding"

    def _read_document_file(self, file_path: str, file_ext: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Read document files (PDF, DOCX, etc.)."""
        try:
            if file_ext == '.pdf':
                return self._read_pdf(file_path)
            elif file_ext == '.docx':
                return self._read_docx(file_path)
            else:
                return None, "error", f"Unsupported document type: {file_ext}"
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return None, "error", f"Document read error: {e}"

    def _read_pdf(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Read PDF file content."""
        try:
            import pypdf

            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                text_parts = []

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {file_path}: {e}")
                        continue

                if text_parts:
                    return '\n\n'.join(text_parts), "text", None
                else:
                    return None, "error", "No text content extracted from PDF"

        except ImportError:
            logger.warning("pypdf not available for PDF reading")
            return None, "error", "PDF support not available (pypdf not installed)"
        except Exception as e:
            return None, "error", f"PDF read error: {e}"

    def _read_docx(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Read DOCX file content."""
        try:
            import mammoth

            with open(file_path, 'rb') as f:
                result = mammoth.extract_raw_text(f)
                if result.value:
                    return result.value, "text", None
                else:
                    return None, "error", "No text content in DOCX file"

        except ImportError:
            logger.warning("mammoth not available for DOCX reading")
            return None, "error", "DOCX support not available (mammoth not installed)"
        except Exception as e:
            return None, "error", f"DOCX read error: {e}"

    def _read_unknown_file(self, file_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Try to read an unknown file type, detecting if it's binary."""
        try:
            # Read a small sample to check if it's binary
            with open(file_path, 'rb') as f:
                sample = f.read(1024)

            # Simple binary detection - if there are many null bytes or non-printable chars
            null_count = sample.count(b'\x00')
            if null_count > len(sample) * 0.1:  # More than 10% null bytes
                return None, "binary", None

            # Try to decode as text
            try:
                # Try UTF-8 first
                with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                    content = f.read()
                return content, "text", None
            except UnicodeDecodeError:
                # Try with error replacement
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                # If too many replacement characters, probably binary
                if content.count('�') > len(content) * 0.05:  # More than 5% replacement chars
                    return None, "binary", None
                return content, "text", None

        except Exception as e:
            return None, "error", f"Unknown file read error: {e}"