import logging
from typing import Optional
from datetime import datetime

from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QCloseEvent, QFont, QFontDatabase, QTextCharFormat, QColor, QPalette
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QHBoxLayout,
                               QPushButton, QLabel, QFrame, QScrollBar)

try:
    from utils import constants
except ImportError:
    class constants_fallback:
        CHAT_FONT_FAMILY = "Courier New"
        CHAT_FONT_SIZE = 9


    constants = constants_fallback
    logging.getLogger(__name__).warning("LlmTerminalWindow: Could not import constants, using fallback values.")

logger = logging.getLogger(__name__)


class LlmTerminalWindow(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("🤖 LLM Communication Terminal")
        self.setObjectName("LlmTerminalWindow")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)

        self._log_text_edit: Optional[QTextEdit] = None
        self._message_count = 0
        self._init_ui()
        self._apply_dark_theme()
        logger.info("LlmTerminalWindow initialized with modern styling.")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar with title and controls
        header_frame = QFrame()
        header_frame.setObjectName("HeaderFrame")
        header_frame.setFixedHeight(50)

        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 10, 15, 10)

        # Title with icon
        title_label = QLabel("🔍 LLM Communication Log")
        title_label.setObjectName("TitleLabel")

        # Connection status indicator
        self.status_label = QLabel("● CONNECTED")
        self.status_label.setObjectName("StatusLabel")

        # Control buttons
        clear_btn = QPushButton("Clear Log")
        clear_btn.setObjectName("ClearButton")
        clear_btn.clicked.connect(self.clear_log)

        export_btn = QPushButton("Export")
        export_btn.setObjectName("ExportButton")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        header_layout.addWidget(clear_btn)
        header_layout.addWidget(export_btn)

        main_layout.addWidget(header_frame)

        # Main terminal area
        self._log_text_edit = QTextEdit()
        self._log_text_edit.setObjectName("TerminalTextEdit")
        self._log_text_edit.setReadOnly(True)

        # Set up monospace font
        font = QFont("JetBrains Mono", 11)  # Modern monospace font
        if not font.exactMatch():
            font = QFont("Consolas", 11)
            if not font.exactMatch():
                font = QFont("Courier New", 11)

        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self._log_text_edit.setFont(font)

        # Custom scrollbar styling will be handled in CSS
        self._log_text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._log_text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        main_layout.addWidget(self._log_text_edit, 1)
        self.setLayout(main_layout)

    def _apply_dark_theme(self):
        """Apply modern dark theme styling"""
        self.setStyleSheet("""
            /* Main window */
            QWidget#LlmTerminalWindow {
                background-color: #1a1a1a;
                color: #e0e0e0;
            }

            /* Header frame */
            QFrame#HeaderFrame {
                background-color: #2d2d2d;
                border-bottom: 1px solid #404040;
            }

            /* Title label */
            QLabel#TitleLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }

            /* Status label */
            QLabel#StatusLabel {
                color: #4ade80;
                font-size: 12px;
                font-weight: bold;
            }

            /* Buttons */
            QPushButton#ClearButton, QPushButton#ExportButton {
                background-color: #404040;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                min-width: 60px;
            }

            QPushButton#ClearButton:hover, QPushButton#ExportButton:hover {
                background-color: #505050;
                border-color: #666666;
            }

            QPushButton#ClearButton:pressed, QPushButton#ExportButton:pressed {
                background-color: #353535;
            }

            /* Terminal text edit */
            QTextEdit#TerminalTextEdit {
                background-color: #0f0f0f;
                color: #e0e0e0;
                border: none;
                padding: 15px;
                selection-background-color: #264f78;
                selection-color: #ffffff;
            }

            /* Scrollbars */
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border: none;
            }

            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }

            QScrollBar:horizontal {
                background-color: #2d2d2d;
                height: 12px;
                border: none;
            }

            QScrollBar::handle:horizontal {
                background-color: #555555;
                border-radius: 6px;
                min-width: 20px;
            }

            QScrollBar::handle:horizontal:hover {
                background-color: #666666;
            }
        """)

    def _get_message_color(self, sender_type: str) -> str:
        """Get color for different message types"""
        colors = {
            "USER": "#60a5fa",  # Blue
            "ASSISTANT": "#34d399",  # Green
            "SYSTEM": "#f59e0b",  # Orange
            "ERROR": "#ef4444",  # Red
            "PLANNER": "#a78bfa",  # Purple
            "CODER": "#06b6d4",  # Cyan
            "TERMINAL": "#6b7280",  # Gray
            "DEBUG": "#9ca3af"  # Light gray
        }
        return colors.get(sender_type.upper(), "#e0e0e0")

    def _get_formatted_timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime("%H:%M:%S")

    @Slot(str)
    def add_log_entry(self, html_text: str):
        """Add a log entry with improved formatting"""
        if not self._log_text_edit:
            logger.warning("LlmTerminalWindow: _log_text_edit is None, cannot add log entry.")
            return

        self._message_count += 1

        # Move cursor to end
        cursor = self._log_text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._log_text_edit.setTextCursor(cursor)

        # Add separator line for new conversations
        if self._message_count > 1:
            separator = f'<div style="border-top: 1px solid #404040; margin: 10px 0;"></div>'
            self._log_text_edit.insertHtml(separator)

        # Parse sender from HTML (basic extraction)
        sender = "SYSTEM"
        if "USER" in html_text.upper():
            sender = "USER"
        elif "ASSISTANT" in html_text.upper() or "AI" in html_text.upper():
            sender = "ASSISTANT"
        elif "ERROR" in html_text.upper():
            sender = "ERROR"
        elif "PLANNER" in html_text.upper():
            sender = "PLANNER"
        elif "CODER" in html_text.upper():
            sender = "CODER"
        elif "TERMINAL" in html_text.upper():
            sender = "TERMINAL"

        # Create styled message header
        color = self._get_message_color(sender)
        timestamp = self._get_formatted_timestamp()

        header_html = f'''
        <div style="margin: 8px 0;">
            <span style="color: {color}; font-weight: bold;">
                ● {sender}
            </span>
            <span style="color: #888888; font-size: 10px; margin-left: 10px;">
                {timestamp}
            </span>
        </div>
        '''

        # Style the content with proper indentation
        content_html = f'''
        <div style="margin-left: 20px; padding: 5px 0; border-left: 2px solid {color}; padding-left: 10px;">
            {html_text}
        </div>
        '''

        self._log_text_edit.insertHtml(header_html)
        self._log_text_edit.insertHtml(content_html)

        # Auto-scroll to bottom
        scrollbar = self._log_text_edit.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())

    @Slot()
    def clear_log(self):
        """Clear the log with confirmation"""
        if self._log_text_edit:
            self._log_text_edit.clear()
            self._message_count = 0

            # Add welcome message
            welcome_html = f'''
            <div style="text-align: center; color: #888888; margin: 50px 0;">
                <h2 style="color: #60a5fa;">🤖 LLM Communication Terminal</h2>
                <p>Ready to monitor AI conversations...</p>
                <p style="font-size: 10px;">Started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            '''
            self._log_text_edit.insertHtml(welcome_html)

            logger.info("LLM Terminal log cleared.")
        else:
            logger.warning("LlmTerminalWindow: _log_text_edit is None, cannot clear log.")

    def update_connection_status(self, connected: bool, backend_name: str = ""):
        """Update the connection status indicator"""
        if connected:
            self.status_label.setText(f"● CONNECTED {backend_name}")
            self.status_label.setStyleSheet("color: #4ade80; font-weight: bold;")
        else:
            self.status_label.setText("● DISCONNECTED")
            self.status_label.setStyleSheet("color: #ef4444; font-weight: bold;")

    def closeEvent(self, event: QCloseEvent):
        logger.debug("LlmTerminalWindow closeEvent: Hiding window.")
        self.hide()
        event.ignore()