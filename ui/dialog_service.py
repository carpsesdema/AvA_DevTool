# ui/dialog_service.py
import logging
from typing import Optional

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QWidget, QDialog, QMessageBox

try:
    from core.event_bus import EventBus
    from core.chat_manager import ChatManager
    from ui.dialogs.llm_terminal_window import LlmTerminalWindow
    from ui.dialogs.personality_dialog import EditPersonalityDialog
    from ui.dialogs.code_viewer_dialog import CodeViewerWindow
    # MODIFICATION: Import the new ProjectRagDialog
    from ui.dialogs.project_rag_dialog import ProjectRagDialog
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
        self._code_viewer_window: Optional[CodeViewerWindow] = None
        self._project_rag_dialog: Optional[ProjectRagDialog] = None  # MODIFICATION: Add instance variable

        self._connect_event_bus_subscriptions_phase1()
        logger.info("DialogService (Phase 1) initialized and connected to EventBus.")

    def _connect_event_bus_subscriptions_phase1(self):
        bus = self._event_bus
        bus.showLlmLogWindowRequested.connect(self.show_llm_terminal_window)
        bus.chatLlmPersonalityEditRequested.connect(self.trigger_edit_personality_dialog)
        bus.viewCodeViewerRequested.connect(lambda: self.show_code_viewer(ensure_creation=True))
        bus.showProjectRagDialogRequested.connect(self.trigger_show_project_rag_dialog)
        # MODIFICATION: Add subscription to show Project RAG Dialog
        # This signal will be emitted by LeftControlPanel's button
        # An alternative would be for LeftControlPanel to call a method on DialogService directly.
        # For now, let's assume an event `showProjectRagDialogRequested` which needs to be added to EventBus.
        # Or, more simply, LeftPanel will call `trigger_show_project_rag_dialog` directly.
        # Let's assume direct call for now, so no new event bus connection here for *triggering* it.

    def show_llm_terminal_window(self, ensure_creation: bool = True) -> Optional[LlmTerminalWindow]:
        logger.debug(f"DialogService: Request to show LLM terminal window (ensure_creation={ensure_creation}).")
        try:
            if self._llm_terminal_window is None and ensure_creation:
                self._llm_terminal_window = LlmTerminalWindow(parent=None)  # Parent None for top-level
                logger.info("DialogService: Created new LlmTerminalWindow instance.")
                if hasattr(self.chat_manager, 'get_llm_communication_logger') and \
                        (llm_logger := self.chat_manager.get_llm_communication_logger()):
                    try:
                        llm_logger.new_terminal_log_entry.disconnect(self._llm_terminal_window.add_log_entry)
                    except (TypeError, RuntimeError):  # Was not connected or already disconnected
                        pass
                    llm_logger.new_terminal_log_entry.connect(self._llm_terminal_window.add_log_entry)
                    logger.info("DialogService: Connected LLM logger to new LlmTerminalWindow.")

            if self._llm_terminal_window:
                self._llm_terminal_window.show()
                self._llm_terminal_window.activateWindow()
                self._llm_terminal_window.raise_()
            return self._llm_terminal_window
        except Exception as e_term:
            logger.error(f"Error showing LlmTerminalWindow: {e_term}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open LLM Terminal:\n{e_term}")
            return None

    def show_code_viewer(self, ensure_creation: bool = True) -> Optional[CodeViewerWindow]:
        logger.debug(f"DialogService: Request to show Code Viewer (ensure_creation={ensure_creation}).")
        try:
            if self._code_viewer_window is None and ensure_creation:
                self._code_viewer_window = CodeViewerWindow(parent=self.parent_window)  # Parent is main window
                logger.info("DialogService: Created new CodeViewerWindow instance.")
                if hasattr(self._code_viewer_window, 'apply_change_requested'):
                    self._code_viewer_window.apply_change_requested.connect(
                        lambda proj_id, rel_fp, content, focus_p:
                        self._event_bus.applyFileChangeRequested.emit(proj_id, rel_fp, content, focus_p)
                    )
                    logger.info("DialogService: Connected CodeViewerWindow.apply_change_requested to EventBus.")
                else:
                    logger.warning(
                        "DialogService: CodeViewerWindow instance does not have 'apply_change_requested' signal.")

            if self._code_viewer_window:
                self._code_viewer_window.show()
                self._code_viewer_window.activateWindow()
                self._code_viewer_window.raise_()
            return self._code_viewer_window
        except Exception as e_cv:
            logger.error(f"Error showing CodeViewerWindow: {e_cv}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open Code Viewer:\n{e_cv}")
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

    # MODIFICATION: New method to show the Project RAG Dialog
    @Slot()
    def trigger_show_project_rag_dialog(self):
        logger.debug("DialogService: Request to show Project RAG File Add dialog.")
        project_manager = self.chat_manager.get_project_manager()
        if not project_manager:
            logger.error("DialogService: ProjectManager not available via ChatManager.")
            QMessageBox.critical(self.parent_window, "Error", "Project manager is not available.")
            return

        current_project = project_manager.get_current_project()
        if not current_project:
            logger.warning("DialogService: No active project to add RAG files to.")
            QMessageBox.information(self.parent_window, "No Active Project",
                                    "Please select or create a project before adding files to its knowledge base.")
            return

        try:
            # We can reuse the dialog instance or create a new one each time.
            # For modal dialogs that don't retain much state beyond the call, creating new is often cleaner.
            self._project_rag_dialog = ProjectRagDialog(
                project_id=current_project.id,
                project_name=current_project.name,
                parent=self.parent_window
            )
            logger.info(f"DialogService: Created/Reused ProjectRagDialog for project '{current_project.name}'.")

            # exec_() is blocking, dialog handles emitting signal if files are confirmed
            self._project_rag_dialog.exec()
            # No need to check result here, dialog emits event on accept.

        except Exception as e_pr_dlg:
            logger.error(f"Error showing ProjectRagDialog: {e_pr_dlg}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error",
                                 f"Could not open Project RAG File Dialog:\n{e_pr_dlg}")

    def close_non_modal_dialogs(self):
        logger.info("DialogService attempting to close non-modal dialogs.")
        if self._llm_terminal_window and self._llm_terminal_window.isVisible():
            try:
                self._llm_terminal_window.close()  # close() on QWidget hides it by default
                logger.debug("  LLM Terminal window close requested.")
            except Exception as e_close_llm:
                logger.error(f"Error closing LlmTerminalWindow: {e_close_llm}")

        if self._code_viewer_window and self._code_viewer_window.isVisible():
            try:
                self._code_viewer_window.close()  # close() on QWidget hides it by default
                logger.debug("  Code Viewer window close requested.")
            except Exception as e_close_cv:
                logger.error(f"Error closing CodeViewerWindow: {e_close_cv}")

        # ProjectRagDialog is modal (uses exec()), so it doesn't need to be in close_non_modal_dialogs.
        # If it were non-modal, we'd add:
        # if self._project_rag_dialog and self._project_rag_dialog.isVisible():
        #     try:
        #         self._project_rag_dialog.close()
        #         logger.debug("  Project RAG Dialog close requested.")
        #     except Exception as e_close_prd:
        #         logger.error(f"Error closing ProjectRagDialog: {e_close_prd}")

