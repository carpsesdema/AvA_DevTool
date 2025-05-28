# ui/dialogs/llm_terminal_window.py - Enhanced with code streaming support
import logging
from typing import Dict, Any, Optional
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit,
                               QPushButton, QLabel, QSizePolicy, QSplitter)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QFont, QTextCursor, QTextDocument

logger = logging.getLogger(__name__)


class LlmTerminalWindow(QDialog):
    """Enhanced LLM Terminal Window with real-time code streaming support"""

    def __init__(self, llm_comm_logger, parent=None):
        super().__init__(parent)
        self._llm_comm_logger = llm_comm_logger

        # NEW: Code streaming state management
        self._streaming_block_cursors: Dict[str, QTextCursor] = {}
        self._streaming_block_elements: Dict[str, Dict[str, Any]] = {}
        # Structure: {block_id: {'pre_cursor_start': QTextCursor, 'code_cursor_start': QTextCursor, 'code_cursor_end': QTextCursor}}

        self._setup_ui()
        self._connect_signals()
        self._load_existing_logs()

        # Auto-scroll tracking
        self._user_scrolled_up = False
        self._scroll_timer = QTimer()
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._check_auto_scroll)

        logger.info("Enhanced LlmTerminalWindow initialized with code streaming support")

    def _setup_ui(self):
        self.setWindowTitle("LLM Communication Terminal")
        self.setModal(False)
        self.resize(1000, 700)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("🤖 LLM Communication Log")
        header_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #e0e0e0;
                padding: 8px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
            }
        """)
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        # Control buttons
        self._clear_button = QPushButton("Clear Log")
        self._clear_button.setStyleSheet("""
            QPushButton {
                background: #404040;
                color: #e0e0e0;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #505050;
                border-color: #707070;
            }
            QPushButton:pressed {
                background: #353535;
            }
        """)
        self._clear_button.clicked.connect(self._clear_log)

        self._auto_scroll_button = QPushButton("Auto-Scroll: ON")
        self._auto_scroll_button.setCheckable(True)
        self._auto_scroll_button.setChecked(True)
        self._auto_scroll_button.setStyleSheet("""
            QPushButton {
                background: #2e7d32;
                color: #e0e0e0;
                border: 1px solid #4caf50;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #388e3c;
                border-color: #66bb6a;
            }
            QPushButton:checked {
                background: #2e7d32;
                border-color: #4caf50;
            }
            QPushButton:!checked {
                background: #404040;
                border-color: #606060;
            }
        """)
        self._auto_scroll_button.toggled.connect(self._toggle_auto_scroll)

        header_layout.addWidget(self._clear_button)
        header_layout.addWidget(self._auto_scroll_button)
        layout.addLayout(header_layout)

        # Main log display
        self._log_text_edit = QTextEdit()
        self._log_text_edit.setReadOnly(True)
        self._log_text_edit.setFont(QFont("JetBrains Mono", 10))

        # Enhanced styling for better code display
        self._log_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #3b3b3b;
                selection-color: #ffffff;
            }
            QScrollBar:vertical {
                background: #2e2e2e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #606060;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #707070;
            }
        """)

        # Monitor scroll position
        scrollbar = self._log_text_edit.verticalScrollBar()
        scrollbar.valueChanged.connect(self._on_scroll_changed)

        layout.addWidget(self._log_text_edit, 1)

        # Status bar
        self._status_label = QLabel("Ready - Monitoring LLM communications")
        self._status_label.setStyleSheet("""
            QLabel {
                color: #b0b0b0;
                font-size: 11px;
                padding: 4px 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._status_label)

    def _connect_signals(self):
        """Connect to LlmCommunicationLogger signals"""
        if self._llm_comm_logger:
            # Regular logging
            self._llm_comm_logger.new_log_entry.connect(self.add_log_entry)

            # NEW: Code streaming signals
            self._llm_comm_logger.code_block_stream_started.connect(self.handle_code_block_stream_started)
            self._llm_comm_logger.code_block_chunk_received.connect(self.handle_code_block_chunk_received)
            self._llm_comm_logger.code_block_stream_finished.connect(self.handle_code_block_stream_finished)

    def _load_existing_logs(self):
        """Load any existing log entries"""
        if self._llm_comm_logger:
            existing_logs = self._llm_comm_logger.get_all_logs()
            for log_entry in existing_logs:
                self.add_log_entry(log_entry)

    @Slot(str)
    def add_log_entry(self, formatted_entry: str):
        """Add a regular log entry to the terminal"""
        cursor = self._log_text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(formatted_entry)

        self._auto_scroll_if_needed()
        self._update_status()

    # NEW: Code streaming slots
    @Slot(str, str)
    def handle_code_block_stream_started(self, block_id: str, language_hint: str):
        """Handle the start of a code streaming session"""
        logger.info(f"Starting code block display for {block_id} ({language_hint})")

        # Get cursor at end of document
        cursor = self._log_text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Store the position before inserting HTML structure
        pre_cursor_start_pos = cursor.position()

        # Create the HTML structure for the code block
        # Using a unique marker that we can find and replace
        marker_id = f"stream-code-{block_id}"
        code_block_html = f"""
<div style="margin: 8px 0;">
<pre style="background-color: #0f0f0f; color: #e0e0e0; padding: 12px; margin: 0; border-radius: 6px; border: 2px solid #404040; white-space: pre-wrap; word-wrap: break-word; font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace; line-height: 1.5; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"><code id="{marker_id}">STREAMING_MARKER_{block_id}</code></pre>
</div>"""

        # Insert the HTML structure
        cursor.insertHtml(code_block_html)

        # Now find the marker text and position cursor there
        document = self._log_text_edit.document()
        marker_text = f"STREAMING_MARKER_{block_id}"

        # Search for the marker
        cursor = document.find(marker_text)
        if cursor.isNull():
            logger.error(f"Could not find streaming marker for block {block_id}")
            return

        # Select and remove the marker text
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        cursor.removeSelectedText()

        # Now cursor is positioned where code should be inserted
        code_cursor_start_pos = cursor.position()

        # Store cursor information for this block
        self._streaming_block_cursors[block_id] = cursor
        self._streaming_block_elements[block_id] = {
            'pre_cursor_start_pos': pre_cursor_start_pos,
            'code_cursor_start_pos': code_cursor_start_pos,
            'code_cursor_end_pos': code_cursor_start_pos,
            'language': language_hint
        }

        self._auto_scroll_if_needed()
        self._update_status(f"Streaming {language_hint} code...")

    @Slot(str, str)
    def handle_code_block_chunk_received(self, block_id: str, plain_text_chunk: str):
        """Handle receiving a chunk of code during streaming"""
        if block_id not in self._streaming_block_cursors:
            logger.warning(f"Received chunk for unknown block_id: {block_id}")
            return

        # Get the cursor for this block
        cursor = self._streaming_block_cursors[block_id]

        # Insert the plain text chunk
        cursor.insertText(plain_text_chunk)

        # Update the end position
        self._streaming_block_elements[block_id]['code_cursor_end_pos'] = cursor.position()

        # Keep cursor position for next chunk
        self._streaming_block_cursors[block_id] = cursor

        self._auto_scroll_if_needed()

        # Update status periodically (not for every chunk to avoid spam)
        if len(plain_text_chunk) > 50 or '\n' in plain_text_chunk:
            total_length = self._streaming_block_elements[block_id]['code_cursor_end_pos'] - \
                           self._streaming_block_elements[block_id]['code_cursor_start_pos']
            language = self._streaming_block_elements[block_id]['language']
            self._update_status(f"Streaming {language} code... ({total_length} chars)")

    @Slot(str, str)
    def handle_code_block_stream_finished(self, block_id: str, final_highlighted_html_content: str):
        """Handle completion of code streaming with syntax highlighting"""
        if block_id not in self._streaming_block_elements:
            logger.warning(f"Received completion for unknown block_id: {block_id}")
            return

        logger.info(f"Finalizing code block {block_id} with syntax highlighting")

        # Get stored cursor positions
        block_info = self._streaming_block_elements[block_id]
        start_pos = block_info['code_cursor_start_pos']
        end_pos = block_info['code_cursor_end_pos']
        language = block_info['language']

        # Create cursor to select all the plain text that was streamed
        replace_cursor = self._log_text_edit.textCursor()
        replace_cursor.setPosition(start_pos)
        replace_cursor.setPosition(end_pos, QTextCursor.MoveMode.KeepAnchor)

        # Remove the plain text
        replace_cursor.removeSelectedText()

        # Insert the highlighted HTML content
        replace_cursor.insertHtml(final_highlighted_html_content)

        # Clean up tracking data
        del self._streaming_block_cursors[block_id]
        del self._streaming_block_elements[block_id]

        self._auto_scroll_if_needed()
        self._update_status(f"Code block completed ({language})")

        # Return to "Ready" status after a short delay
        QTimer.singleShot(3000, lambda: self._update_status("Ready - Monitoring LLM communications"))

    def _auto_scroll_if_needed(self):
        """Auto-scroll to bottom if user hasn't scrolled up manually"""
        if self._auto_scroll_button.isChecked() and not self._user_scrolled_up:
            scrollbar = self._log_text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _on_scroll_changed(self, value):
        """Track if user has scrolled up manually"""
        scrollbar = self._log_text_edit.verticalScrollBar()
        max_value = scrollbar.maximum()

        # Consider user has scrolled up if they're not at the bottom
        # Add some tolerance for rounding errors
        self._user_scrolled_up = (value < max_value - 5)

        # Reset scroll flag after a delay if user scrolls back to bottom
        if not self._user_scrolled_up:
            self._scroll_timer.start(1000)

    def _check_auto_scroll(self):
        """Re-enable auto-scroll if user is at bottom"""
        scrollbar = self._log_text_edit.verticalScrollBar()
        if scrollbar.value() >= scrollbar.maximum() - 5:
            self._user_scrolled_up = False

    def _toggle_auto_scroll(self, enabled):
        """Toggle auto-scroll feature"""
        self._auto_scroll_button.setText(f"Auto-Scroll: {'ON' if enabled else 'OFF'}")
        if enabled:
            self._user_scrolled_up = False
            self._auto_scroll_if_needed()

    def _clear_log(self):
        """Clear the log display and logger"""
        self._log_text_edit.clear()
        if self._llm_comm_logger:
            self._llm_comm_logger.clear_logs()

        # Clean up any active streaming state
        self._streaming_block_cursors.clear()
        self._streaming_block_elements.clear()

        self._update_status("Log cleared")
        logger.info("LLM terminal log cleared")

    def _update_status(self, message: str = None):
        """Update the status label"""
        if message:
            self._status_label.setText(message)
        else:
            # Show number of active streams if any
            active_streams = len(self._streaming_block_cursors)
            if active_streams > 0:
                self._status_label.setText(f"Ready - {active_streams} active code stream(s)")
            else:
                self._status_label.setText("Ready - Monitoring LLM communications")

    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up any active streams
        if self._streaming_block_cursors:
            logger.info(f"Cleaning up {len(self._streaming_block_cursors)} active code streams on close")
            self._streaming_block_cursors.clear()
            self._streaming_block_elements.clear()

        event.accept()

    def showEvent(self, event):
        """Handle window show event"""
        super().showEvent(event)
        # Scroll to bottom when window is shown
        self._auto_scroll_if_needed()