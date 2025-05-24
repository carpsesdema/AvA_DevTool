import logging
from typing import Optional

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QWidget, QDialog, QMessageBox

try:
    from core.event_bus import EventBus
    from core.chat_manager import ChatManager
    from ui.dialogs.llm_terminal_window import LlmTerminalWindow
    from ui.dialogs.personality_dialog import EditPersonalityDialog
except ImportError as e_ds:
    logging.getLogger(__name__).critical(f"Critical import error in DialogService: {e_ds}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class DialogService(QObject):
    def __init__(self, parent_window: QWidget, chat_manager: ChatManager, event_bus: EventBus):
        super().__init__(parent_window)
        self.parent_window = parent_window

        if not isinstance(chat_manager, ChatManager):
            logger.critical("DialogService initialized with invalid ChatManager instance.")
            raise ValueError("DialogService requires a valid ChatManager instance.")
        if not isinstance(event_bus, EventBus):
            logger.critical("DialogService initialized with invalid EventBus instance.")
            raise ValueError("DialogService requires a valid EventBus instance.")

        self.chat_manager = chat_manager
        self._event_bus = event_bus

        self._llm_terminal_window: Optional[LlmTerminalWindow] = None

        self._connect_event_bus_subscriptions_phase1()
        logger.info("DialogService (Phase 1) initialized and connected to EventBus.")

    def _connect_event_bus_subscriptions_phase1(self):
        bus = self._event_bus
        bus.showLlmLogWindowRequested.connect(self.show_llm_terminal_window)
        bus.chatLlmPersonalityEditRequested.connect(self.trigger_edit_personality_dialog)

    @Slot()
    def show_llm_terminal_window(self, ensure_creation: bool = True) -> Optional[LlmTerminalWindow]:
        logger.debug(f"DialogService: Request to show LLM terminal window (ensure_creation={ensure_creation}).")
        try:
            if self._llm_terminal_window is None and ensure_creation:
                self._llm_terminal_window = LlmTerminalWindow(parent=None)
                logger.info("DialogService: Created new LlmTerminalWindow instance.")

            if self._llm_terminal_window:
                self._llm_terminal_window.show()
                self._llm_terminal_window.activateWindow()
                self._llm_terminal_window.raise_()
            return self._llm_terminal_window
        except Exception as e_term:
            logger.error(f"Error showing LlmTerminalWindow: {e_term}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open LLM Terminal:\n{e_term}")
            return None

    @Slot()
    def trigger_edit_personality_dialog(self) -> None:
        logger.debug(f"DialogService: Request to show Edit Personality dialog.")
        try:
            current_prompt = self.chat_manager.get_current_chat_personality()
            active_chat_backend_id = self.chat_manager.get_current_active_chat_backend_id()

            dialog = EditPersonalityDialog(current_prompt, parent=self.parent_window)
            dialog_result = dialog.exec()

            if dialog_result == QDialog.DialogCode.Accepted:
                new_prompt_text = dialog.get_prompt_text()
                logger.info(
                    f"DialogService: Personality dialog accepted. New prompt for '{active_chat_backend_id}': '{new_prompt_text[:50]}...'")
                self._event_bus.chatLlmPersonalitySubmitted.emit(new_prompt_text, active_chat_backend_id)
            else:
                logger.info(f"DialogService: Edit Personality dialog cancelled or closed (Result: {dialog_result}).")
        except Exception as e_pers_dlg:
            logger.error(f"Error showing EditPersonalityDialog: {e_pers_dlg}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error",
                                 f"Could not open Personality Editor:\n{e_pers_dlg}")

    def close_non_modal_dialogs(self):
        logger.info("DialogService attempting to close non-modal dialogs.")
        if self._llm_terminal_window:
            try:
                self._llm_terminal_window.close()
                logger.debug("  LLM Terminal window close requested.")
            except Exception as e_close_llm:
                logger.error(f"Error closing LlmTerminalWindow: {e_close_llm}")

