# services/llm_communication_logger.py - Enhanced with code streaming
import logging
import uuid
import html
from datetime import datetime
from typing import Dict, List, Optional, Any
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

try:
    import pygments
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import HtmlFormatter
    from pygments.util import ClassNotFound

    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    logger.warning("Pygments not available. Code highlighting will be disabled.")


class LlmCommunicationLogger(QObject):
    # Regular logging signals
    new_log_entry = Signal(str)  # formatted_log_entry

    # NEW: Code streaming signals
    code_block_stream_started = Signal(str, str)  # block_id, language_hint
    code_block_chunk_received = Signal(str, str)  # block_id, plain_text_chunk
    code_block_stream_finished = Signal(str, str)  # block_id, final_highlighted_html_content

    # Styled content configuration
    STYLED_CONTENT_SENDERS = {
        "GEMINI_CHAT_DEFAULT RESPONSE", "GPT_CHAT_DEFAULT RESPONSE", "OLLAMA_CHAT_DEFAULT RESPONSE",
        "PACC_V2:SEQ_START_ROBUST", "PACC_V2:MULTI_FILE_COMPLETE", "RAG_SCAN_GLOBAL",
        "AUTONOMOUS_CODING_REQUEST", "AUTONOMOUS_CODING_ERROR", "AUTONOMOUS_CODING_EXCEPTION",
        "CODE_EXTRACTION", "CODE_EXTRACTION_ERROR",
        "SYSTEM_CODE_STREAM"  # NEW: For code streaming messages
    }

    HTML_SENDER_STYLES = {
        "GEMINI_CHAT_DEFAULT RESPONSE": {"color": "#4fc3f7", "symbol": "🧠"},
        "GPT_CHAT_DEFAULT RESPONSE": {"color": "#81c784", "symbol": "🤖"},
        "OLLAMA_CHAT_DEFAULT RESPONSE": {"color": "#ffb74d", "symbol": "🦙"},
        "PACC_V2:SEQ_START_ROBUST": {"color": "#e57373", "symbol": "🚀"},
        "PACC_V2:MULTI_FILE_COMPLETE": {"color": "#aed581", "symbol": "✅"},
        "RAG_SCAN_GLOBAL": {"color": "#ba68c8", "symbol": "🔍"},
        "AUTONOMOUS_CODING_REQUEST": {"color": "#64b5f6", "symbol": "⚙️"},
        "AUTONOMOUS_CODING_ERROR": {"color": "#ef5350", "symbol": "❌"},
        "AUTONOMOUS_CODING_EXCEPTION": {"color": "#ff5722", "symbol": "💥"},
        "CODE_EXTRACTION": {"color": "#26a69a", "symbol": "📦"},
        "CODE_EXTRACTION_ERROR": {"color": "#ff7043", "symbol": "⚠️"},
        "SYSTEM_CODE_STREAM": {"color": "#9575cd", "symbol": "📜"}  # NEW: Code streaming style
    }

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._log_entries: List[str] = []

        # NEW: Code streaming state management
        self._active_code_streams: Dict[str, Dict[str, Any]] = {}
        # Structure: {block_id: {'language': str, 'buffer': List[str]}}

        logger.info("Enhanced LlmCommunicationLogger initialized with code streaming support")

    def log_message(self, sender: str, message: str) -> None:
        """Log a regular message (non-code streaming)"""
        if not sender or not message:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Apply styling if sender is in styled list
        if sender in self.STYLED_CONTENT_SENDERS:
            style_info = self.HTML_SENDER_STYLES.get(sender, {"color": "#ffffff", "symbol": "💬"})
            color = style_info["color"]
            symbol = style_info["symbol"]

            # Enhanced styling for better readability
            formatted_entry = (
                f'<div style="margin: 4px 0; padding: 6px 12px; background: rgba(255,255,255,0.05); '
                f'border-left: 3px solid {color}; border-radius: 4px;">'
                f'<span style="color: #888; font-size: 11px; font-family: monospace;">[{timestamp}]</span> '
                f'<span style="color: {color}; font-weight: bold;">{symbol} {sender}:</span><br/>'
                f'<span style="color: #e0e0e0; margin-left: 16px; display: block; margin-top: 4px;">{html.escape(message)}</span>'
                f'</div>'
            )
        else:
            # Regular message formatting
            formatted_entry = (
                f'<div style="margin: 2px 0; padding: 4px 8px;">'
                f'<span style="color: #888; font-size: 11px;">[{timestamp}]</span> '
                f'<span style="color: #b0b0b0; font-weight: bold;">{html.escape(sender)}:</span> '
                f'<span style="color: #e0e0e0;">{html.escape(message)}</span>'
                f'</div>'
            )

        self._log_entries.append(formatted_entry)
        self.new_log_entry.emit(formatted_entry)

    # NEW: Code streaming methods
    def start_streaming_code_block(self, language_hint: str = "python") -> str:
        """
        Initiate a new code streaming session.

        Args:
            language_hint: The programming language for syntax highlighting

        Returns:
            Unique block_id for this streaming session
        """
        block_id = uuid.uuid4().hex

        # Initialize streaming state
        self._active_code_streams[block_id] = {
            'language': language_hint,
            'buffer': []
        }

        # Log the start of streaming
        self.log_message("SYSTEM_CODE_STREAM", f"Starting code block streaming ({language_hint})...")

        # Emit signal for terminal to prepare UI
        self.code_block_stream_started.emit(block_id, language_hint)

        logger.info(f"Started code streaming block {block_id} with language {language_hint}")
        return block_id

    def stream_code_chunk(self, block_id: str, chunk: str) -> None:
        """
        Add a chunk of raw code to an active stream.

        Args:
            block_id: The streaming session ID
            chunk: Raw code text chunk
        """
        if block_id not in self._active_code_streams:
            logger.warning(f"Attempted to stream chunk to unknown block_id: {block_id}")
            return

        # Add chunk to buffer
        self._active_code_streams[block_id]['buffer'].append(chunk)

        # Emit signal for terminal to display chunk immediately
        self.code_block_chunk_received.emit(block_id, chunk)

        logger.debug(f"Streamed chunk to block {block_id}: {len(chunk)} chars")

    def finish_streaming_code_block(self, block_id: str) -> None:
        """
        Finalize an active code stream with syntax highlighting.

        Args:
            block_id: The streaming session ID to finalize
        """
        if block_id not in self._active_code_streams:
            logger.warning(f"Attempted to finish unknown block_id: {block_id}")
            return

        stream_info = self._active_code_streams[block_id]
        language = stream_info['language']
        code_chunks = stream_info['buffer']

        # Concatenate all chunks into complete code
        complete_code = ''.join(code_chunks)

        # Generate syntax highlighted HTML
        highlighted_html = self._highlight_code_block_html(complete_code, language)

        # Log completion
        self.log_message("SYSTEM_CODE_STREAM", f"Code block completed ({len(complete_code)} chars)")

        # Emit signal for terminal to replace plain text with highlighted version
        self.code_block_stream_finished.emit(block_id, highlighted_html)

        # Clean up
        del self._active_code_streams[block_id]

        logger.info(f"Finished code streaming block {block_id}")

    def _highlight_code_block_html(self, code: str, language_hint: str) -> str:
        """
        Generate syntax-highlighted HTML content for code.

        Args:
            code: Complete code string
            language_hint: Programming language for syntax highlighting

        Returns:
            HTML content suitable for insertion inside a <code> tag
        """
        if not PYGMENTS_AVAILABLE or not code.strip():
            # Fallback: just escape HTML and preserve whitespace
            return html.escape(code).replace('\n', '<br/>')

        try:
            # Try to get lexer by language hint
            try:
                lexer = get_lexer_by_name(language_hint, stripall=True)
            except ClassNotFound:
                # Fallback: try to guess the lexer
                try:
                    lexer = guess_lexer(code, stripall=True)
                except ClassNotFound:
                    # Ultimate fallback: no highlighting
                    return html.escape(code).replace('\n', '<br/>')

            # Configure formatter for inline styles (no CSS classes)
            formatter = HtmlFormatter(
                style='material',  # Dark-friendly theme
                noclasses=True,  # Use inline styles
                nowrap=True  # Don't wrap in <pre> tags
            )

            # Generate highlighted HTML
            highlighted = highlight(code, lexer, formatter)

            # The formatter with nowrap=True should give us just the spans
            # But let's ensure we clean up any unwanted wrapper tags
            if highlighted.startswith('<div') and highlighted.endswith('</div>'):
                # Remove outer div wrapper if present
                start_idx = highlighted.find('>') + 1
                end_idx = highlighted.rfind('<')
                highlighted = highlighted[start_idx:end_idx]

            return highlighted

        except Exception as e:
            logger.warning(f"Code highlighting failed for language '{language_hint}': {e}")
            # Fallback to escaped HTML
            return html.escape(code).replace('\n', '<br/>')

    def get_all_logs(self) -> List[str]:
        """Get all logged entries"""
        return self._log_entries.copy()

    def clear_logs(self) -> None:
        """Clear all logged entries"""
        self._log_entries.clear()
        logger.info("LlmCommunicationLogger logs cleared")

    def get_active_streams_count(self) -> int:
        """Get the number of active code streams (for debugging)"""
        return len(self._active_code_streams)

    def cancel_all_streams(self) -> None:
        """Cancel all active code streams (cleanup method)"""
        cancelled_count = len(self._active_code_streams)
        self._active_code_streams.clear()
        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} active code streams")
            self.log_message("SYSTEM_CODE_STREAM", f"Cancelled {cancelled_count} active code streams")