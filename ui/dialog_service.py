# ui/dialog_service.py - Fixed version with better code viewer management
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
    from ui.dialogs.project_rag_dialog import ProjectRagDialog
    # NEW: Add update dialog imports
    from ui.dialogs.update_dialog import UpdateDialog
    from services.update_service import UpdateInfo
    from utils import constants
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
        self._project_rag_dialog: Optional[ProjectRagDialog] = None
        self._update_dialog: Optional[UpdateDialog] = None  # NEW: Add update dialog instance

        self._connect_event_bus_subscriptions_phase1()
        logger.info("DialogService (Phase 1) initialized and connected to EventBus.")

    def _connect_event_bus_subscriptions_phase1(self):
        bus = self._event_bus
        bus.showLlmLogWindowRequested.connect(self.show_llm_terminal_window)
        bus.chatLlmPersonalityEditRequested.connect(self.trigger_edit_personality_dialog)
        bus.viewCodeViewerRequested.connect(lambda: self.show_code_viewer(ensure_creation=True))
        bus.showProjectRagDialogRequested.connect(self.trigger_show_project_rag_dialog)

        # NEW: Connect update-related signals
        bus.updateAvailable.connect(self.show_update_dialog)
        bus.noUpdateAvailable.connect(self._handle_no_update_available)
        bus.updateCheckFailed.connect(self._handle_update_check_failed)

    def show_llm_terminal_window(self, ensure_creation: bool = True) -> Optional[LlmTerminalWindow]:
        logger.debug(f"DialogService: Request to show LLM terminal window (ensure_creation={ensure_creation}).")
        try:
            if self._llm_terminal_window is None and ensure_creation:
                llm_logger = self.chat_manager.get_llm_communication_logger()
                if llm_logger:
                    self._llm_terminal_window = LlmTerminalWindow(llm_logger, parent=None)
                    logger.info("DialogService: Created new LlmTerminalWindow with logger.")
                else:
                    logger.error("DialogService: Could not create LlmTerminalWindow - no logger available")
                    return None
                logger.info("DialogService: Created new LlmTerminalWindow instance.")

                # Connect the LLM communication logger
                if hasattr(self.chat_manager, 'get_llm_communication_logger'):
                    llm_logger = self.chat_manager.get_llm_communication_logger()
                    if llm_logger and hasattr(llm_logger, 'new_terminal_log_entry'):
                        try:
                            # Disconnect any existing connections first
                            llm_logger.new_terminal_log_entry.disconnect()
                        except (TypeError, RuntimeError):
                            pass  # No existing connections

                        # Connect to the new terminal window
                        llm_logger.new_terminal_log_entry.connect(self._llm_terminal_window.add_log_entry)
                        logger.info("DialogService: Connected LLM logger to new LlmTerminalWindow.")
                    else:
                        logger.warning("DialogService: LLM logger not available or missing signal.")

            if self._llm_terminal_window:
                self._llm_terminal_window.show()
                self._llm_terminal_window.activateWindow()
                self._llm_terminal_window.raise_()
                logger.debug("DialogService: LLM terminal window shown and activated.")

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

                # Connect the apply change signal
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
                logger.debug("DialogService: Code viewer window shown and activated.")

            return self._code_viewer_window

        except Exception as e_cv:
            logger.error(f"Error showing CodeViewerWindow: {e_cv}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not open Code Viewer:\n{e_cv}")
            return None

    def get_or_create_code_viewer(self) -> Optional[CodeViewerWindow]:
        """Get existing code viewer or create new one - used by file display functionality"""
        return self.show_code_viewer(ensure_creation=True)

    def display_file_in_code_viewer(self, filename: str, content: str, project_id: Optional[str] = None,
                                    focus_prefix: Optional[str] = None) -> bool:
        """Display a file in the code viewer - handles creation and display"""
        try:
            code_viewer = self.get_or_create_code_viewer()
            if not code_viewer:
                logger.error("DialogService: Could not get/create code viewer for file display")
                return False

            # Update or add the file
            code_viewer.update_or_add_file(
                filename=filename,
                content=content,
                is_ai_modification=True,
                original_content=None,
                project_id_for_apply=project_id,
                focus_prefix_for_apply=focus_prefix
            )

            # Ensure the code viewer is visible and focused
            code_viewer.show()
            code_viewer.activateWindow()
            code_viewer.raise_()

            logger.info(f"DialogService: Successfully displayed file '{filename}' in code viewer")
            return True

        except Exception as e:
            logger.error(f"DialogService: Error displaying file '{filename}' in code viewer: {e}", exc_info=True)
            return False

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
            self._project_rag_dialog = ProjectRagDialog(
                project_id=current_project.id,
                project_name=current_project.name,
                parent=self.parent_window
            )
            logger.info(f"DialogService: Created/Reused ProjectRagDialog for project '{current_project.name}'.")
            self._project_rag_dialog.exec()

        except Exception as e_pr_dlg:
            logger.error(f"Error showing ProjectRagDialog: {e_pr_dlg}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error",
                                 f"Could not open Project RAG File Dialog:\n{e_pr_dlg}")

    # NEW: Update dialog methods
    @Slot(object)
    def show_update_dialog(self, update_info: UpdateInfo):
        """Show the update dialog when an update is available"""
        logger.info(f"Showing update dialog for version {update_info.version}")

        try:
            # Close existing dialog if open
            if self._update_dialog and self._update_dialog.isVisible():
                self._update_dialog.close()

            # Create new update dialog
            self._update_dialog = UpdateDialog(update_info, parent=self.parent_window)

            # Connect dialog signals to event bus
            self._update_dialog.download_requested.connect(
                lambda info: self._event_bus.updateDownloadRequested.emit(info)
            )
            self._update_dialog.install_requested.connect(
                lambda path: self._event_bus.updateInstallRequested.emit(path)
            )
            self._update_dialog.restart_requested.connect(
                lambda: self._event_bus.applicationRestartRequested.emit()
            )

            # Connect progress signals
            self._event_bus.updateProgress.connect(self._update_dialog.update_progress)
            self._event_bus.updateStatusChanged.connect(self._update_dialog.update_status)
            self._event_bus.updateDownloaded.connect(self._update_dialog.download_completed)
            self._event_bus.updateDownloadFailed.connect(self._update_dialog.download_failed)

            # Show the dialog
            self._update_dialog.show()
            self._update_dialog.activateWindow()
            self._update_dialog.raise_()

        except Exception as e:
            logger.error(f"Error showing update dialog: {e}", exc_info=True)
            QMessageBox.critical(self.parent_window, "Dialog Error", f"Could not show update dialog:\n{e}")

    @Slot()
    def _handle_no_update_available(self):
        """Handle when no update is available"""
        logger.info("No update available")
        QMessageBox.information(
            self.parent_window,
            "No Updates Available",
            f"You are running the latest version of {constants.APP_NAME}.\n\n"
            f"Current version: {constants.APP_VERSION}"
        )

    @Slot(str)
    def _handle_update_check_failed(self, error_message: str):
        """Handle update check failure"""
        logger.error(f"Update check failed: {error_message}")
        QMessageBox.warning(
            self.parent_window,
            "Update Check Failed",
            f"Could not check for updates:\n{error_message}\n\n"
            "Please check your internet connection and try again."
        )

    def close_non_modal_dialogs(self):
        logger.info("DialogService attempting to close non-modal dialogs.")
        if self._llm_terminal_window and self._llm_terminal_window.isVisible():
            try:
                self._llm_terminal_window.close()
                logger.debug("  LLM Terminal window close requested.")
            except Exception as e_close_llm:
                logger.error(f"Error closing LlmTerminalWindow: {e_close_llm}")

        if self._code_viewer_window and self._code_viewer_window.isVisible():
            try:
                self._code_viewer_window.close()
                logger.debug("  Code Viewer window close requested.")
            except Exception as e_close_cv:
                logger.error(f"Error closing CodeViewerWindow: {e_close_cv}")

        # NEW: Close update dialog if open
        if self._update_dialog and self._update_dialog.isVisible():
            try:
                self._update_dialog.close()
                logger.debug("  Update dialog close requested.")
            except Exception as e_close_update:
                logger.error(f"Error closing UpdateDialog: {e_close_update}")