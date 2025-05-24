import logging
from typing import Optional, List, Dict, Any

from PySide6.QtCore import QSize, Slot, QTimer, Signal as pyqtSignal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSizePolicy
)

try:
    from utils import constants
    from ui.multiline_input_widget import MultilineInputWidget
except ImportError as e_cib:
    logging.getLogger(__name__).critical(f"Critical import error in ChatInputBar: {e_cib}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatInputBar(QWidget):
    sendMessageRequested = pyqtSignal()

    ACTION_BUTTON_SIZE = QSize(28, 28)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatInputBar")

        self._multiline_input: Optional[MultilineInputWidget] = None
        self._send_button: Optional[QPushButton] = None

        self._is_sending_blocked = False
        self._is_busy_external = False
        self._is_explicitly_disabled = False

        self._init_ui_phase1()
        self._connect_signals_phase1()
        self._update_button_state_phase1()

    def _init_ui_phase1(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 2, 5, 2)
        main_layout.setSpacing(5)

        self._multiline_input = MultilineInputWidget(self)
        main_layout.addWidget(self._multiline_input, 1)

        self._send_button = QPushButton("Send", self)
        self._send_button.setObjectName("SendButton")
        send_button_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1)
        self._send_button.setFont(send_button_font)
        self._send_button.setToolTip("Send message (Enter)\nAdd newline (Shift+Enter)")
        self._send_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        min_height = self.ACTION_BUTTON_SIZE.height()
        if hasattr(self._multiline_input, '_min_height'):  # Match MultilineInput's min height
            min_height = max(min_height, self._multiline_input._min_height)  # type: ignore
        self._send_button.setMinimumHeight(min_height)

        main_layout.addWidget(self._send_button)

    def _connect_signals_phase1(self):
        if self._multiline_input:
            self._multiline_input.sendMessageRequested.connect(self._on_send_requested_phase1)
            self._multiline_input.textChanged.connect(self._update_button_state_phase1)

        if self._send_button:
            self._send_button.clicked.connect(self._on_send_requested_phase1)

    @Slot()
    def _on_send_requested_phase1(self):
        if self._is_sending_blocked:
            return

        if self._is_explicitly_disabled or self._is_busy_external:
            return

        text_to_send = self.get_text()
        if not text_to_send:
            return

        self._is_sending_blocked = True
        self.sendMessageRequested.emit()
        self.clear_text()
        QTimer.singleShot(100, lambda: setattr(self, '_is_sending_blocked', False))

    @Slot(bool)
    def handle_busy_state(self, is_busy: bool):
        if self._is_busy_external == is_busy: return
        self._is_busy_external = is_busy
        self._update_button_state_phase1()
        if self._multiline_input:
            self._multiline_input.set_enabled(not self._is_explicitly_disabled and not self._is_busy_external)

    @Slot()
    def _update_button_state_phase1(self):
        if self._send_button:
            can_send = (not self._is_explicitly_disabled and
                        not self._is_busy_external and
                        bool(self.get_text()))
            self._send_button.setEnabled(can_send)

    def get_text(self) -> str:
        return self._multiline_input.get_text() if self._multiline_input else ""

    def clear_text(self):
        if self._multiline_input: self._multiline_input.clear_text()
        self._update_button_state_phase1()

    def set_focus(self):
        if self._multiline_input: self._multiline_input.set_focus()

    def set_enabled_overall(self, enabled: bool):
        self._is_explicitly_disabled = not enabled
        self.handle_busy_state(self._is_busy_external)

