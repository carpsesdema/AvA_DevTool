import logging
from typing import Optional

from PySide6.QtGui import QFont, QCloseEvent
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox, QLabel, QWidget
)

try:
    from utils import constants
except ImportError:
    class constants_fallback:
        CHAT_FONT_FAMILY = "Arial"
        CHAT_FONT_SIZE = 10


    constants = constants_fallback
    logging.getLogger(__name__).warning("EditPersonalityDialog: Could not import constants, using fallback values.")

logger = logging.getLogger(__name__)


class EditPersonalityDialog(QDialog):
    def __init__(self, current_prompt: Optional[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Configure AI Persona / System Prompt")
        self.setObjectName("PersonalityDialog")
        self.setMinimumSize(550, 350)
        self.setModal(True)

        self._prompt_edit: Optional[QTextEdit] = None
        self._init_widgets(current_prompt)
        self._init_layout()
        self._connect_signals()
        logger.info("EditPersonalityDialog initialized.")

    def _init_widgets(self, current_prompt: Optional[str]):
        try:
            dialog_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE)
            label_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1)
        except AttributeError:  # Fallback if constants not fully loaded
            dialog_font = QFont("Arial", 10)
            label_font = QFont("Arial", 9)
            logger.warning("EditPersonalityDialog: Using fallback fonts due to constants issue.")

        self.info_label = QLabel(
            "Define the system prompt or personality for the main Chat LLM.\n"
            "This guides its behavior, tone, and expertise. Leave empty for default."
        )
        self.info_label.setFont(label_font)
        self.info_label.setWordWrap(True)

        self._prompt_edit = QTextEdit()
        self._prompt_edit.setObjectName("PersonalityPromptEdit")
        self._prompt_edit.setFont(dialog_font)
        self._prompt_edit.setPlaceholderText("e.g., You are a helpful assistant specializing in Python...")
        self._prompt_edit.setPlainText(current_prompt or "")
        self._prompt_edit.setAcceptRichText(False)
        self._prompt_edit.setMinimumHeight(150)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

    def _init_layout(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        layout.addWidget(self.info_label)
        if self._prompt_edit:
            layout.addWidget(self._prompt_edit, 1)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def _connect_signals(self):
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def get_prompt_text(self) -> str:
        return self._prompt_edit.toPlainText().strip() if self._prompt_edit else ""

    def showEvent(self, event: QCloseEvent):  # QCloseEvent is for closeEvent, showEvent takes QShowEvent
        super().showEvent(event)  # type: ignore
        if self._prompt_edit:
            self._prompt_edit.setFocus()
        logger.debug("EditPersonalityDialog shown and focus set.")

    def exec(self) -> int:
        logger.debug("EditPersonalityDialog exec() called.")
        result = super().exec()
        logger.debug(f"EditPersonalityDialog exec() finished with result: {result}")
        return result

    def closeEvent(self, event: QCloseEvent):
        logger.debug(f"EditPersonalityDialog closeEvent. Dialog result: {self.result()}")
        super().closeEvent(event)

