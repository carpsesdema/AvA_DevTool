import logging
import os
import sys
from typing import Optional, List, Any

from PySide6.QtCore import Qt, Slot, QTimer, QEvent
from PySide6.QtGui import QFont, QIcon, QCloseEvent, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QApplication, QMessageBox,
    QLabel, QStyle
)

try:
    from core.event_bus import EventBus
    from core.chat_manager import ChatManager
    from core.models import ChatMessage
    from ui.left_panel import LeftControlPanel
    from ui.dialog_service import DialogService
    from ui.chat_display_area import ChatDisplayArea
    from ui.chat_input_bar import ChatInputBar
    from ui.loading_overlay import LoadingOverlay
    from utils import constants
    from core.chat_message_state_handler import ChatMessageStateHandler
    from services.project_service import Project, ChatSession
except ImportError as e_main_window:
    logging.basicConfig(level=logging.DEBUG)
    logging.critical(f"CRITICAL IMPORT ERROR in main_window.py: {e_main_window}", exc_info=True)
    try:
        _dummy_app = QApplication(sys.argv) if QApplication.instance() is None else QApplication.instance()
        QMessageBox.critical(None, "Import Error",
                             f"Failed to import critical UI components:\n{e_main_window}\nCheck logs and installation.")
    except Exception as msg_e:
        logging.critical(f"Failed to show import error message box: {msg_e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


class MainWindow(QWidget):
    def __init__(self, chat_manager: ChatManager, app_base_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        logger.info("MainWindow initializing...")
        if not isinstance(chat_manager, ChatManager):
            logger.critical("MainWindow requires a valid ChatManager instance.")
            raise TypeError("MainWindow requires a valid ChatManager instance.")

        self.chat_manager = chat_manager
        self._project_manager = chat_manager.get_project_manager()
        self.app_base_path = app_base_path
        self._event_bus = EventBus.get_instance()

        self.left_panel: Optional[LeftControlPanel] = None
        self.status_label: Optional[QLabel] = None
        self._status_clear_timer: Optional[QTimer] = None
        self.dialog_service: Optional[DialogService] = None
        self.active_chat_display_area: Optional[ChatDisplayArea] = None
        self.active_chat_input_bar: Optional[ChatInputBar] = None
        self._chat_message_state_handler: Optional[ChatMessageStateHandler] = None
        self._loading_overlay: Optional[LoadingOverlay] = None

        self._current_base_status_text: str = "Status: Initializing..."
        self._current_base_status_color: str = "#abb2bf"

        try:
            self.dialog_service = DialogService(self, self.chat_manager, self._event_bus)
        except Exception as e_ds:
            logger.critical(f"Failed to initialize DialogService in MainWindow: {e_ds}", exc_info=True)
            QApplication.quit()
            return

        self._init_ui()
        self._apply_styles()
        self._connect_signals_and_event_bus()
        self._connect_project_manager_signals_to_ui()
        self._setup_window_properties()
        self._init_loading_overlay()
        logger.info("MainWindow initialized successfully.")

    def _setup_window_properties(self):
        self.setWindowTitle(constants.APP_NAME) # type: ignore
        try:
            icon_filename = getattr(constants, 'APP_ICON_FILENAME', "Synchat.ico") # type: ignore
            icon_path_candidates = [
                os.path.join(constants.ASSETS_PATH, icon_filename), # type: ignore
                os.path.join(os.path.dirname(sys.executable), icon_filename) if getattr(sys, 'frozen', False) else "",
                os.path.join(self.app_base_path, "assets", icon_filename)
            ]
            app_icon_path = ""
            for p in icon_path_candidates:
                if p and os.path.exists(p):
                    app_icon_path = p
                    break

            if app_icon_path:
                self.setWindowIcon(QIcon(app_icon_path))
                logger.info(f"Application icon set from: {app_icon_path}")
            else:
                logger.warning(f"Application icon not found. Candidates: {icon_path_candidates}")
                std_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
                if not std_icon.isNull(): self.setWindowIcon(std_icon)
        except Exception as e_icon:
            logger.error(f"Error setting window icon: {e_icon}", exc_info=True)
        self.update_window_title()

    def _init_loading_overlay(self):
        try:
            self._loading_overlay = LoadingOverlay(parent=self)
            logger.info("Loading overlay initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize loading overlay: {e}")
            self._loading_overlay = None

    def _init_ui(self):
        main_hbox_layout = QHBoxLayout(self)
        main_hbox_layout.setContentsMargins(0, 0, 0, 0)
        main_hbox_layout.setSpacing(0)
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.setObjectName("MainSplitter")
        main_splitter.setHandleWidth(1)

        try:
            self.left_panel = LeftControlPanel(chat_manager=self.chat_manager, parent=self)
            self.left_panel.setObjectName("LeftPanel")
            self.left_panel.setMinimumWidth(260)
            self.left_panel.setMaximumWidth(400)
        except Exception as e_lcp:
            logger.critical(f"CRITICAL ERROR creating LeftControlPanel: {e_lcp}", exc_info=True)
            QMessageBox.critical(self, "Initialization Error", f"Failed to create Left Panel:\n{e_lcp}")
            QApplication.quit()
            return

        right_panel_widget = QWidget(self)
        right_panel_widget.setObjectName("RightPanelContainer")
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(5, 5, 5, 5)
        right_panel_layout.setSpacing(5)
        right_panel_widget.setMinimumWidth(450)

        self.active_chat_display_area = ChatDisplayArea(parent=right_panel_widget)
        if self.active_chat_display_area and self.active_chat_display_area.chat_item_delegate:
            self.active_chat_display_area.chat_item_delegate.setView(self.active_chat_display_area.chat_list_view)

        self.active_chat_input_bar = ChatInputBar(parent=right_panel_widget)

        right_panel_layout.addWidget(self.active_chat_display_area, 1)
        right_panel_layout.addWidget(self.active_chat_input_bar)

        status_bar_widget = QWidget(self)
        status_bar_widget.setObjectName("StatusBarWidget")
        status_bar_layout = QHBoxLayout(status_bar_widget)
        status_bar_layout.setContentsMargins(8, 3, 8, 3)
        status_bar_layout.setSpacing(10)
        self.status_label = QLabel("Status: Initializing...", self)
        self.status_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 2)) # type: ignore
        self.status_label.setObjectName("StatusLabel")
        status_bar_layout.addWidget(self.status_label, 1)
        right_panel_layout.addWidget(status_bar_widget)

        main_splitter.addWidget(self.left_panel)
        main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([270, 730])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_hbox_layout.addWidget(main_splitter)
        self.setLayout(main_hbox_layout)

        self._chat_message_state_handler = ChatMessageStateHandler(self._event_bus, parent=self)

    def _apply_styles(self):
        logger.debug("MainWindow applying styles...")
        try:
            stylesheet_path = ""
            for path_candidate in constants.STYLE_PATHS_TO_CHECK: # type: ignore
                if os.path.exists(path_candidate):
                    stylesheet_path = path_candidate
                    break
            if stylesheet_path:
                with open(stylesheet_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
                logger.info(f"Loaded stylesheet from: {stylesheet_path}")
            else:
                logger.warning(f"Stylesheet not found. Applying basic default.")
                self.setStyleSheet(
                    "QWidget { background-color: #282c34; color: #abb2bf; } QLabel#StatusLabel { color: #A0A0A0; }")
        except Exception as e_style:
            logger.error(f"Error loading/applying stylesheet: {e_style}", exc_info=True)
            self.setStyleSheet("QWidget { background-color: #333; color: #EEE; }")

    def _connect_signals_and_event_bus(self):
        if not all([self.chat_manager, self.left_panel, self.dialog_service,
                    self.active_chat_display_area, self.active_chat_input_bar]):
            logger.error("MainWindow: One or more critical UI components are None. Cannot connect signals.")
            return

        bus = self._event_bus
        if self.active_chat_input_bar:
            self.active_chat_input_bar.sendMessageRequested.connect(
                lambda: bus.userMessageSubmitted.emit(
                    self.active_chat_input_bar.get_text() if self.active_chat_input_bar else "",
                    self.active_chat_input_bar.get_attached_image_data() if hasattr(self.active_chat_input_bar, # type: ignore
                                                                                    'get_attached_image_data') else []
                )
            )

        if self.active_chat_display_area:
            self.active_chat_display_area.textCopied.connect(
                lambda text, color: bus.uiTextCopied.emit(text, color)
            )

        bus.uiStatusUpdateGlobal.connect(self.update_status)
        bus.uiErrorGlobal.connect(self._handle_error_event)
        bus.uiInputBarBusyStateChanged.connect(self._handle_input_bar_busy_state_change)
        bus.backendConfigurationChanged.connect(self._handle_backend_configuration_changed_event)

        bus.showLoader.connect(self._show_loading_overlay)
        bus.hideLoader.connect(self._hide_loading_overlay)
        bus.updateLoaderMessage.connect(self._update_loading_message)

        bus.modificationFileReadyForDisplay.connect(self._handle_code_file_update_event)

        bus.newMessageAddedToHistory.connect(self._handle_new_message_added_to_history)
        bus.activeSessionHistoryCleared.connect(self._handle_active_session_cleared)
        bus.activeSessionHistoryLoaded.connect(self._handle_active_session_history_loaded)
        bus.messageChunkReceivedForSession.connect(self._handle_message_chunk_for_session)
        bus.messageFinalizedForSession.connect(self._handle_message_finalized_for_session)

        shortcut_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        shortcut_escape.activated.connect(self._handle_escape_key_pressed)

    def _connect_project_manager_signals_to_ui(self):
        if not self._project_manager:
            logger.warning("MW: ProjectManager not available for signal connection.")
            return

        logger.debug("MW: Connecting ProjectManager signals for MainWindow specific UI updates.")
        self._project_manager.projectSwitched.connect(self._handle_project_switched_ui_update)
        self._project_manager.sessionSwitched.connect(self._handle_session_switched_ui_update)
        self._project_manager.projectDeleted.connect(self.update_window_title)

    @Slot(str)
    def _show_loading_overlay(self, message: str):
        if self._loading_overlay:
            self._loading_overlay.show_loading(message)
            logger.debug(f"Showing loading overlay: {message}")
        else:
            logger.warning("Loading overlay not available")

    @Slot()
    def _hide_loading_overlay(self):
        if self._loading_overlay:
            self._loading_overlay.hide_loading()
            logger.debug("Hiding loading overlay")

    @Slot(str)
    def _update_loading_message(self, message: str):
        if self._loading_overlay and self._loading_overlay.isVisible():
            self._loading_overlay.update_message(message)
            logger.debug(f"Updated loading message: {message}")

    @Slot(str, str)
    def _handle_code_file_update_event(self, filename: str, content: str):
        logger.info(f"MainWindow: Received code update for '{filename}' via EventBus. Delegating to DialogService.")
        if self.dialog_service:
            project_id_context = self.chat_manager.get_current_project_id() if self.chat_manager else None

            focus_prefix_context = None
            if self._project_manager and project_id_context:
                focus_prefix_context = self._project_manager.get_project_files_dir(project_id_context)
            else:
                focus_prefix_context = os.path.join(os.getcwd(), "ava_generated_projects", "unknown_project")
                os.makedirs(focus_prefix_context, exist_ok=True)

            success = self.dialog_service.display_file_in_code_viewer(
                filename,
                content,
                project_id=project_id_context,
                focus_prefix=focus_prefix_context
            )
            if success:
                logger.info(f"MainWindow: Successfully requested DialogService to display '{filename}'.")
            else:
                logger.error(f"MainWindow: DialogService failed to display '{filename}'.")
                self.update_status(f"Error displaying {filename} in code viewer", "#e06c75", True, 5000)
        else:
            logger.error("MainWindow: DialogService not available to handle code file update.")
            self.update_status("Dialog service for code viewer not available", "#e06c75", True, 3000)

    @Slot(str)
    def _handle_project_switched_ui_update(self, project_id: str):
        logger.info(f"MW: UI Update for project switched (from PM signal): {project_id}")
        self.update_window_title()

    @Slot(str, str)
    def _handle_session_switched_ui_update(self, project_id: str, session_id: str):
        logger.info(f"MW: UI Update for session switched (from PM signal): P:{project_id} S:{session_id}")
        self.update_window_title()

    @Slot(str, str, ChatMessage) # type: ignore
    def _handle_new_message_added_to_history(self, project_id: str, session_id: str, message: ChatMessage): # type: ignore
        current_pid = self.chat_manager.get_current_project_id() if self.chat_manager else None
        current_sid = self.chat_manager.get_current_session_id() if self.chat_manager else None

        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            self.active_chat_display_area.add_message_to_model(project_id, session_id, message)
        else:
            logger.debug(
                f"MW: newMessageAddedToHistory for non-active P/S: {project_id}/{session_id}. Ignored by active display.")

    @Slot(str, str)
    def _handle_active_session_cleared(self, project_id: str, session_id: str):
        current_pid = self.chat_manager.get_current_project_id() if self.chat_manager else None
        current_sid = self.chat_manager.get_current_session_id() if self.chat_manager else None

        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            logger.info(f"MW: Clearing display for active session P:{project_id}/S:{session_id}")
            self.active_chat_display_area.clear_model_display(project_id, session_id)
        else:
            logger.debug(
                f"MW: activeSessionHistoryCleared for non-active P/S: {project_id}/{session_id}. Ignored by active display.")

    @Slot(str, str, list)
    def _handle_active_session_history_loaded(self, project_id: str, session_id: str, history: List[ChatMessage]): # type: ignore
        current_pid = self.chat_manager.get_current_project_id() if self.chat_manager else None
        current_sid = self.chat_manager.get_current_session_id() if self.chat_manager else None

        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            logger.info(f"MW: Loading history for active session P:{project_id}/S:{session_id} into display area.")
            self.active_chat_display_area.set_current_context(project_id, session_id)
            self.active_chat_display_area.load_history_into_model(project_id, session_id, history)

            if self._chat_message_state_handler and self.active_chat_display_area.get_model():
                logger.info(f"MW: Registering model for P:{project_id}/S:{session_id} with ChatMessageStateHandler.")
                self._chat_message_state_handler.register_model_for_project_session(
                    project_id, session_id, self.active_chat_display_area.get_model() # type: ignore
                )
            else:
                logger.warning(
                    f"MW: Could not register model for P:{project_id}/S:{session_id}. CMSH or model missing.")
        else:
            logger.debug(
                f"MW: activeSessionHistoryLoaded for P/S ({project_id}/{session_id}) which is not current active ({current_pid}/{current_sid}). Ignored by active display.")

    @Slot(str, str, str, str)
    def _handle_message_chunk_for_session(self, project_id: str, session_id: str, request_id: str, chunk_text: str):
        current_pid = self.chat_manager.get_current_project_id() if self.chat_manager else None
        current_sid = self.chat_manager.get_current_session_id() if self.chat_manager else None
        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            self.active_chat_display_area.append_chunk_to_message_by_id(request_id, chunk_text)

    @Slot(str, str, str, ChatMessage, dict, bool) # type: ignore
    def _handle_message_finalized_for_session(self, project_id: str, session_id: str, request_id: str,
                                              final_message_obj: ChatMessage, usage_stats_dict: dict, is_error: bool): # type: ignore
        current_pid = self.chat_manager.get_current_project_id() if self.chat_manager else None
        current_sid = self.chat_manager.get_current_session_id() if self.chat_manager else None
        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            self.active_chat_display_area.finalize_message_by_id(request_id, final_message_obj, is_error)

    @Slot(str, str, bool, int)
    def update_status(self, message: str, color: str, is_temporary: bool = False, duration_ms: int = 3000):
        if self.status_label is None: return
        self._current_base_status_text = message
        self._current_base_status_color = color
        self._refresh_full_status_display()

        if self._status_clear_timer:
            self._status_clear_timer.stop()
            self._status_clear_timer = None

        if is_temporary:
            self._status_clear_timer = QTimer(self)
            self._status_clear_timer.setSingleShot(True)
            self._status_clear_timer.timeout.connect(self._clear_temporary_status)
            self._status_clear_timer.start(duration_ms)

    def _refresh_full_status_display(self):
        if not self.status_label: return
        final_status_text = self._current_base_status_text
        self.status_label.setText(final_status_text)
        self.status_label.setStyleSheet(f"QLabel#StatusLabel {{ color: {self._current_base_status_color}; }}")

    def _clear_temporary_status(self):
        if self._status_clear_timer:
            self._status_clear_timer.stop()
            self._status_clear_timer = None

        if self.chat_manager:
            if self.chat_manager.is_api_ready():
                model_name = self.chat_manager.get_model_for_backend(
                    self.chat_manager.get_current_active_chat_backend_id()) or "Unknown Model"
                self.update_status(f"Ready. Using {model_name}", "#98c379", is_temporary=False)
            else:
                last_error = "Unknown configuration error"
                if hasattr(self.chat_manager, '_backend_coordinator') and self.chat_manager._backend_coordinator: # type: ignore
                    last_error = self.chat_manager._backend_coordinator.get_last_error_for_backend( # type: ignore
                        self.chat_manager.get_current_active_chat_backend_id()) or "Backend config issue"
                self.update_status(f"Backend not configured: {last_error}", "#FFCC00", is_temporary=False)
        else:
            self.update_status("Ready", "#98c379", is_temporary=False)

    @Slot(str, bool)
    def _handle_error_event(self, error_message: str, is_critical: bool):
        self.update_status(f"Error: {error_message[:100]}...", "#FF6B6B", True, 7000)
        if is_critical:
            QMessageBox.critical(self, "Critical Application Error", error_message)

    @Slot(bool)
    def _handle_input_bar_busy_state_change(self, is_input_bar_busy: bool):
        if self.active_chat_input_bar:
            self.active_chat_input_bar.handle_busy_state(is_input_bar_busy)
        if self.left_panel and self.chat_manager:
            api_ready = self.chat_manager.is_api_ready()
            rag_ready = self.chat_manager.is_rag_ready()
            self.left_panel.set_enabled_state(
                is_api_ready=api_ready,
                is_busy=is_input_bar_busy,
                is_rag_ready=rag_ready
            )

    @Slot(str, str, bool, list)
    def _handle_backend_configuration_changed_event(self, backend_id: str, model_name: str, is_configured: bool,
                                                    available_models):
        self.update_window_title()
        if self.left_panel and self.chat_manager and backend_id == self.chat_manager.get_current_active_chat_backend_id():
            self.left_panel.update_personality_tooltip(
                active=bool(self.chat_manager.get_current_chat_personality()))

        if is_configured:
            logger.info(
                f"MW: Backend '{backend_id}' configured for model '{model_name}'. Status updated by ChatManager.")
        else:
            logger.warning(
                f"MW: Backend '{backend_id}' not configured for model '{model_name}'. Status updated by ChatManager.")

        if self.chat_manager and backend_id == self.chat_manager.get_current_active_chat_backend_id():
            if is_configured:
                self.update_status(f"Ready. Using {model_name}", "#98c379", False)
            else:
                err = "Unknown reason"
                if hasattr(self.chat_manager, '_backend_coordinator') and self.chat_manager._backend_coordinator: # type: ignore
                    err = self.chat_manager._backend_coordinator.get_last_error_for_backend( # type: ignore
                        backend_id) or "Config error"
                self.update_status(f"Chat LLM Error: {err}", "#FF6B6B", False)

    def _handle_escape_key_pressed(self):
        if self.chat_manager and self.chat_manager.is_overall_busy():
            if hasattr(self.chat_manager,
                       '_current_llm_request_id') and self.chat_manager._current_llm_request_id: # type: ignore
                request_to_cancel = self.chat_manager._current_llm_request_id # type: ignore
                if hasattr(self.chat_manager, '_backend_coordinator') and self.chat_manager._backend_coordinator: # type: ignore
                    self.chat_manager._backend_coordinator.cancel_current_task( # type: ignore
                        request_id=request_to_cancel)
                    self.update_status("Attempting to cancel AI response...", "#e5c07b", True, 2000)
            else:
                if hasattr(self.chat_manager, '_backend_coordinator') and self.chat_manager._backend_coordinator and self.chat_manager._backend_coordinator.is_any_backend_busy(): # type: ignore
                    self.chat_manager._backend_coordinator.cancel_current_task(None) # type: ignore
                    self.update_status("Attempting to cancel ongoing AI tasks...", "#e5c07b", True, 2000)

    def update_window_title(self):
        base_title = constants.APP_NAME # type: ignore
        details = []
        current_project: Optional[Project] = None # type: ignore
        current_session: Optional[ChatSession] = None # type: ignore

        if self._project_manager:
            current_project = self._project_manager.get_current_project()
            current_session = self._project_manager.get_current_session()

        if current_project:
            details.append(f"Project: {current_project.name[:25]}") # type: ignore
        if current_session:
            details.append(f"Session: {current_session.name[:25]}") # type: ignore

        if self.chat_manager:
            active_backend_id = self.chat_manager.get_current_active_chat_backend_id()
            model_name = self.chat_manager.get_model_for_backend(active_backend_id)
            if model_name:
                model_short = model_name.split('/')[-1].split(':')[-1]
                if "gemini" in model_short: model_short = model_short.replace("-preview-05-20", "").replace("-latest",
                                                                                                            "")
                details.append(f"LLM: {model_short[:20]}")
            if self.chat_manager.get_current_chat_personality():
                details.append("Persona")

        self.setWindowTitle(f"{base_title} - [{', '.join(details)}]" if details else base_title)

    def resizeEvent(self, event: QEvent): # type: ignore
        super().resizeEvent(event)

        if self._loading_overlay and self._loading_overlay.isVisible():
            self._loading_overlay.resize(self.size())

    def closeEvent(self, event: QCloseEvent):
        logger.info("MainWindow closeEvent triggered. Performing cleanup...")

        if self._loading_overlay:
            self._loading_overlay.hide_loading()

        if self.dialog_service and hasattr(self.dialog_service, 'close_non_modal_dialogs'):
            self.dialog_service.close_non_modal_dialogs()
        if self.chat_manager and hasattr(self.chat_manager, 'cleanup_phase1'):
            self.chat_manager.cleanup_phase1()

        if self._chat_message_state_handler and self.chat_manager:
            pid = self.chat_manager.get_current_project_id()
            sid = self.chat_manager.get_current_session_id()
            if pid and sid and self.active_chat_display_area and self.active_chat_display_area.get_model():
                self._chat_message_state_handler.unregister_model_for_project_session(pid, sid)
        event.accept()

    def showEvent(self, event: QEvent):
        super().showEvent(event)
        if self.active_chat_input_bar:
            QTimer.singleShot(100, self.active_chat_input_bar.set_focus) # type: ignore
        QTimer.singleShot(150, self._clear_temporary_status)
        self.update_window_title()