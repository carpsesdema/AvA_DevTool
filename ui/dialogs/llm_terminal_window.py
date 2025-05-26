import logging
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtGui import QCloseEvent, QFont, QFontDatabase
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

try:
    from utils import constants
except ImportError:
    class constants_fallback:
        CHAT_FONT_FAMILY = "Courier New"  # Fallback font
        CHAT_FONT_SIZE = 9


    constants = constants_fallback
    logging.getLogger(__name__).warning("LlmTerminalWindow: Could not import constants, using fallback values.")

logger = logging.getLogger(__name__)


class LlmTerminalWindow(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("LLM Communication Log")
        self.setObjectName("LlmTerminalWindow")
        self.setMinimumSize(700, 500)

        self._log_text_edit: Optional[QTextEdit] = None
        self._init_ui()
        logger.info("LlmTerminalWindow initialized.")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        self._log_text_edit = QTextEdit()
        self._log_text_edit.setObjectName("LlmLogTextEdit")
        self._log_text_edit.setReadOnly(True)

        try:
            log_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            log_font.setPointSize(constants.CHAT_FONT_SIZE - 1)
        except AttributeError:  # Fallback if constants not fully loaded
            log_font = QFont("Courier New", 9)
            logger.warning("LlmTerminalWindow: Using fallback font due to constants issue.")

        self._log_text_edit.setFont(log_font)

        main_layout.addWidget(self._log_text_edit, 1)
        self.setLayout(main_layout)

    @Slot(str)
    def add_log_entry(self, html_text: str):
        if self._log_text_edit:
            # FIX: Use insertHtml instead of appendHtml (which doesn't exist)
            cursor = self._log_text_edit.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self._log_text_edit.setTextCursor(cursor)
            self._log_text_edit.insertHtml(html_text)
            self._log_text_edit.insertHtml("<br>")  # Add line break

            # Auto-scroll to bottom
            scrollbar = self._log_text_edit.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())
        else:
            logger.warning("LlmTerminalWindow: _log_text_edit is None, cannot add log entry.")

    @Slot()
    def clear_log(self):
        if self._log_text_edit:
            self._log_text_edit.clear()
            logger.info("LLM Terminal log cleared.")
        else:
            logger.warning("LlmTerminalWindow: _log_text_edit is None, cannot clear log.")

    def closeEvent(self, event: QCloseEvent):
        logger.debug("LlmTerminalWindow closeEvent: Hiding window.")
        self.hide()
        event.ignore()