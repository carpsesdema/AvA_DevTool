# services/chunking_service.py
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Service for chunking text documents into smaller pieces for RAG processing.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        """
        Initialize the chunking service.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"ChunkingService initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def chunk_document(self, content: str, source_id: str, file_ext: str = "") -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces.

        Args:
            content: The text content to chunk
            source_id: Identifier for the source (usually file path)
            file_ext: File extension to help determine chunking strategy

        Returns:
            List of chunk dictionaries with 'content' and 'metadata' keys
        """
        if not content or not content.strip():
            logger.warning(f"Empty content provided for chunking: {source_id}")
            return []

        chunks = []

        # For Python files, try to chunk by logical boundaries (functions, classes)
        if file_ext.lower() == '.py':
            chunks = self._chunk_python_code(content, source_id)
        else:
            # For other files, use simple text chunking
            chunks = self._chunk_text(content, source_id)

        logger.debug(f"Chunked {source_id} into {len(chunks)} pieces")
        return chunks

    def _chunk_python_code(self, content: str, source_id: str) -> List[Dict[str, Any]]:
        """
        Chunk Python code by trying to preserve logical boundaries.
        """
        lines = content.splitlines()
        chunks = []
        current_chunk_lines = []
        current_line_num = 1
        chunk_start_line = 1

        for i, line in enumerate(lines):
            current_chunk_lines.append(line)

            # Check if we should create a chunk
            current_chunk_text = '\n'.join(current_chunk_lines)

            # If chunk is getting large and we're at a logical boundary, create chunk
            if (len(current_chunk_text) >= self.chunk_size and
                    (line.strip() == '' or
                     line.startswith('def ') or
                     line.startswith('class ') or
                     line.startswith('import ') or
                     line.startswith('from '))):

                if current_chunk_lines:
                    chunk_dict = {
                        'content': current_chunk_text,
                        'metadata': {
                            'source': source_id,
                            'chunk_index': len(chunks),
                            'start_line': chunk_start_line,
                            'end_line': current_line_num,
                            'filename': source_id.split('/')[-1] if '/' in source_id else source_id
                        }
                    }
                    chunks.append(chunk_dict)

                # Start new chunk with overlap
                overlap_lines = max(0, self.chunk_overlap // 50)  # Rough estimate of lines
                current_chunk_lines = current_chunk_lines[-overlap_lines:] if overlap_lines > 0 else []
                chunk_start_line = max(1, current_line_num - overlap_lines + 1)

            current_line_num += 1

        # Add remaining content as final chunk
        if current_chunk_lines:
            final_chunk_text = '\n'.join(current_chunk_lines)
            if final_chunk_text.strip():
                chunk_dict = {
                    'content': final_chunk_text,
                    'metadata': {
                        'source': source_id,
                        'chunk_index': len(chunks),
                        'start_line': chunk_start_line,
                        'end_line': current_line_num - 1,
                        'filename': source_id.split('/')[-1] if '/' in source_id else source_id
                    }
                }
                chunks.append(chunk_dict)

        return chunks

    def _chunk_text(self, content: str, source_id: str) -> List[Dict[str, Any]]:
        """
        Chunk regular text content.
        """
        chunks = []
        start_pos = 0
        chunk_index = 0

        while start_pos < len(content):
            # Calculate end position
            end_pos = start_pos + self.chunk_size

            # If this isn't the last chunk, try to break at a natural boundary
            if end_pos < len(content):
                # Look for sentence or paragraph breaks within the overlap distance
                search_start = max(start_pos, end_pos - self.chunk_overlap)

                # Try to find a good break point
                for break_char in ['\n\n', '. ', '\n', ' ']:
                    break_pos = content.rfind(break_char, search_start, end_pos)
                    if break_pos > search_start:
                        end_pos = break_pos + len(break_char)
                        break

            chunk_text = content[start_pos:end_pos]

            if chunk_text.strip():
                # Calculate approximate line numbers
                lines_before = content[:start_pos].count('\n')
                lines_in_chunk = chunk_text.count('\n')

                chunk_dict = {
                    'content': chunk_text,
                    'metadata': {
                        'source': source_id,
                        'chunk_index': chunk_index,
                        'start_line': lines_before + 1,
                        'end_line': lines_before + lines_in_chunk + 1,
                        'filename': source_id.split('/')[-1] if '/' in source_id else source_id
                    }
                }
                chunks.append(chunk_dict)
                chunk_index += 1

            # Move to next chunk with overlap
            start_pos = max(start_pos + 1, end_pos - self.chunk_overlap)

            # Prevent infinite loop
            if start_pos >= len(content):
                break

        return chunks