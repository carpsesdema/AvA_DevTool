import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QFontMetrics, QTextOption, QKeyEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

try:
    from utils import constants
except ImportError as e_miw:
    logging.getLogger(__name__).critical(f"Critical import error in MultilineInputWidget: {e_miw}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class MultilineInputWidget(QWidget):
    sendMessageRequested = Signal()
    textChanged = Signal()

    MIN_LINES = 1
    MAX_LINES = 6
    LINE_PADDING = 8

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("MultilineInputWidget")

        self._text_edit: Optional[QTextEdit] = None
        self._min_height: int = 30
        self._max_height: int = 150

        self._init_ui()
        self._calculate_height_limits()
        self._connect_signals()
        self._update_height()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._text_edit = QTextEdit(self)
        self._text_edit.setObjectName("UserInputTextEdit")
        self._text_edit.setAcceptRichText(False)
        self._text_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)

        try:
            font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE)
            self._text_edit.setFont(font)
        except Exception as e_font:
            logger.error(f"Error setting font for MultilineInputWidget: {e_font}")

        layout.addWidget(self._text_edit)
        self.setLayout(layout)

    def _calculate_height_limits(self):
        if not self._text_edit: return
        try:
            fm = QFontMetrics(self._text_edit.font())
            line_height = fm.height()
            doc_margin = self._text_edit.document().documentMargin() if self._text_edit.document() else 4.0
            frame_width = self._text_edit.frameWidth() * 2

            min_base_content_height = line_height * self.MIN_LINES
            max_base_content_height = line_height * self.MAX_LINES

            vertical_padding_allowance = self.LINE_PADDING + int(doc_margin * 2) + frame_width

            self._min_height = min_base_content_height + vertical_padding_allowance
            self._max_height = max_base_content_height + vertical_padding_allowance
        except Exception as e_calc:
            logger.error(f"Error calculating height limits for MultilineInput: {e_calc}. Using defaults.")
            font_metrics_fallback = QFontMetrics(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE))
            self._min_height = (font_metrics_fallback.height() * self.MIN_LINES) + self.LINE_PADDING
            self._max_height = (font_metrics_fallback.height() * self.MAX_LINES) + self.LINE_PADDING

    def _connect_signals(self):
        if self._text_edit:
            self._text_edit.textChanged.connect(self.textChanged.emit)
            self._text_edit.textChanged.connect(self._update_height)

    @Slot()
    def _update_height(self):
        if not self._text_edit or not self._text_edit.document(): return

        viewport_width = self._text_edit.viewport().width()
        effective_width = viewport_width if viewport_width > 0 else self.width()

        if effective_width > 0:
            self._text_edit.document().setTextWidth(effective_width)
        else:
            self._text_edit.document().setTextWidth(100)

        doc_height = self._text_edit.document().size().height()

        doc_margin = self._text_edit.document().documentMargin()
        frame_width = self._text_edit.frameWidth() * 2
        vertical_padding_allowance = self.LINE_PADDING + int(doc_margin * 2) + frame_width

        target_height = int(doc_height + vertical_padding_allowance)
        clamped_height = max(self._min_height, min(target_height, self._max_height))

        if self.height() != clamped_height:
            self.setFixedHeight(clamped_height)

    def keyPressEvent(self, event: QKeyEvent):
        if not self._text_edit:
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()

        is_enter = key in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        is_shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if is_enter and not is_shift_pressed:
            self.sendMessageRequested.emit()
            event.accept()
        elif is_enter and is_shift_pressed:
            super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def get_text(self) -> str:
        return self._text_edit.toPlainText().strip() if self._text_edit else ""

    def clear_text(self):
        if self._text_edit:
            self._text_edit.clear()

    def set_focus(self):
        if self._text_edit:
            self._text_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_enabled(self, enabled: bool):
        if self._text_edit:
            self._text_edit.setEnabled(enabled)

    def setPlainText(self, text: str):
        if self._text_edit:
            self._text_edit.setPlainText(text)
