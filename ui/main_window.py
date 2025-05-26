# ui/main_window.py
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
    from utils import constants
    from core.chat_message_state_handler import ChatMessageStateHandler
    from services.project_service import Project, ChatSession # For type hint
except ImportError as e_main_window:
    logging.basicConfig(level=logging.DEBUG)
    logging.critical(f"CRITICAL IMPORT ERROR in main_window.py: {e_main_window}", exc_info=True)
    try:
        _dummy_app = QApplication(sys.argv) if QApplication.instance() is None else QApplication.instance()
        QMessageBox.critical(None, "Import Error",
                             f"Failed to import critical UI components:\n{e_main_window}\nCheck installation and paths.")
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
        logger.info("MainWindow initialized successfully.")

    def _setup_window_properties(self):
        self.setWindowTitle(constants.APP_NAME)
        try:
            app_icon_path = os.path.join(constants.ASSETS_PATH, constants.APP_ICON_FILENAME if hasattr(constants,
                                                                                                       'APP_ICON_FILENAME') else "Synchat.ico")
            if os.path.exists(app_icon_path):
                self.setWindowIcon(QIcon(app_icon_path))
            else:
                logger.warning(f"Application icon not found at: {app_icon_path}.")
                std_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
                if not std_icon.isNull():
                    self.setWindowIcon(std_icon)
        except Exception as e_icon:
            logger.error(f"Error setting window icon: {e_icon}", exc_info=True)
        self.update_window_title()

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
        self.status_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 2))
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
            for path_candidate in constants.STYLE_PATHS_TO_CHECK:
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
                    self.active_chat_input_bar.get_attached_image_data() if hasattr(self.active_chat_input_bar,
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
        self._project_manager.projectDeleted.connect(self.update_window_title) # Also update title if project is deleted


    @Slot(str, str)
    def _handle_code_file_update_event(self, filename: str, content: str):
        logger.info(f"MainWindow: Received code update for '{filename}' via EventBus.")
        if self.dialog_service:
            code_viewer = self.dialog_service.show_code_viewer(ensure_creation=True)
            if code_viewer:
                project_id = self.chat_manager.get_current_project_id() if self.chat_manager else "unknown_project"
                focus_prefix = self._project_manager.get_project_files_dir(
                    project_id) if self._project_manager and project_id != "unknown_project" else self.app_base_path

                code_viewer.update_or_add_file(
                    filename, content, is_ai_modification=True, original_content=None,
                    project_id_for_apply=project_id, focus_prefix_for_apply=focus_prefix
                )
                logger.info(
                    f"MainWindow: Successfully added/updated '{filename}' in CodeViewer for project '{project_id}'")
            else:
                logger.error("MainWindow: CodeViewerDialog instance could not be obtained/created.")
        else:
            logger.error("MainWindow: DialogService not available to show CodeViewer.")

    @Slot(str)
    def _handle_project_switched_ui_update(self, project_id: str):
        logger.info(f"MW: UI Update for project switched (from PM signal): {project_id}")
        self.update_window_title()

    @Slot(str, str)
    def _handle_session_switched_ui_update(self, project_id: str, session_id: str):
        logger.info(f"MW: UI Update for session switched (from PM signal): P:{project_id} S:{session_id}")
        self.update_window_title()

    @Slot(str, str, ChatMessage)
    def _handle_new_message_added_to_history(self, project_id: str, session_id: str, message: ChatMessage):
        current_pid = self.chat_manager.get_current_project_id()
        current_sid = self.chat_manager.get_current_session_id()
        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            self.active_chat_display_area.add_message_to_model(project_id, session_id, message)
        else:
            logger.debug(
                f"MW: newMessageAddedToHistory for non-active P/S: {project_id}/{session_id}. Ignored by active display.")

    @Slot(str, str)
    def _handle_active_session_cleared(self, project_id: str, session_id: str):
        current_pid = self.chat_manager.get_current_project_id()
        current_sid = self.chat_manager.get_current_session_id()
        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            logger.info(f"MW: Clearing display for active session P:{project_id}/S:{session_id}")
            self.active_chat_display_area.clear_model_display(project_id, session_id)
        else:
            logger.debug(
                f"MW: activeSessionHistoryCleared for non-active P/S: {project_id}/{session_id}. Ignored by active display.")

    @Slot(str, str, list)
    def _handle_active_session_history_loaded(self, project_id: str, session_id: str, history: List[ChatMessage]):
        current_pid = self.chat_manager.get_current_project_id()
        current_sid = self.chat_manager.get_current_session_id()

        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            logger.info(f"MW: Loading history for active session P:{project_id}/S:{session_id} into display area.")
            self.active_chat_display_area.set_current_context(project_id, session_id)
            self.active_chat_display_area.load_history_into_model(project_id, session_id, history)

            if self._chat_message_state_handler and self.active_chat_display_area.get_model():
                logger.info(f"MW: Registering model for P:{project_id}/S:{session_id} with ChatMessageStateHandler.")
                self._chat_message_state_handler.register_model_for_project_session(
                    project_id, session_id, self.active_chat_display_area.get_model()
                )
            else:
                logger.warning(f"MW: Could not register model for P:{project_id}/S:{session_id}. CMSH or model missing.")
        else:
            logger.debug(
                f"MW: activeSessionHistoryLoaded for P/S ({project_id}/{session_id}) which is not current active ({current_pid}/{current_sid}). Ignored by active display.")

    @Slot(str, str, str, str)
    def _handle_message_chunk_for_session(self, project_id: str, session_id: str, request_id: str, chunk_text: str):
        current_pid = self.chat_manager.get_current_project_id()
        current_sid = self.chat_manager.get_current_session_id()
        if self.active_chat_display_area and project_id == current_pid and session_id == current_sid:
            self.active_chat_display_area.append_chunk_to_message_by_id(request_id, chunk_text)

    @Slot(str, str, str, ChatMessage, dict, bool)
    def _handle_message_finalized_for_session(self, project_id: str, session_id: str, request_id: str,
                                              final_message_obj: ChatMessage, usage_stats_dict: dict, is_error: bool):
        current_pid = self.chat_manager.get_current_project_id()
        current_sid = self.chat_manager.get_current_session_id()
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
            self._status_clear_timer.deleteLater()
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
            self._status_clear_timer.deleteLater()
            self._status_clear_timer = None

        if self.chat_manager and self.chat_manager.is_api_ready():
            self.update_status(
                f"Ready. Using {self.chat_manager.get_model_for_backend(self.chat_manager.get_current_active_chat_backend_id())}",
                "#98c379", is_temporary=False)
        elif self.chat_manager:
            last_error = self.chat_manager._backend_coordinator.get_last_error_for_backend(
                self.chat_manager.get_current_active_chat_backend_id()) if self.chat_manager._backend_coordinator else "Unknown"
            self.update_status(
                f"Backend not configured: {last_error}", "#FFCC00", is_temporary=False)
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
        if self.left_panel:
            api_ready = self.chat_manager.is_api_ready() if self.chat_manager else False
            rag_ready = self.chat_manager.is_rag_ready() if self.chat_manager else False
            # FIX: Use correct parameter names for set_enabled_state
            self.left_panel.set_enabled_state(
                is_api_ready=api_ready,
                is_busy=is_input_bar_busy,
                is_rag_ready=rag_ready
            )

    @Slot(str, str, bool, list)
    def _handle_backend_configuration_changed_event(self, backend_id: str, model_name: str, is_configured: bool,
                                                    available_models: list):
        self.update_window_title()
        if self.left_panel and self.chat_manager and backend_id == self.chat_manager.get_current_active_chat_backend_id():
            self.left_panel.update_personality_tooltip(
                active=bool(self.chat_manager.get_current_chat_personality()))

        if is_configured:
            logger.info(f"MW: Backend '{backend_id}' configured. Status updated via ChatManager.")
        else:
            logger.warning(f"MW: Backend '{backend_id}' not configured. Status updated via ChatManager.")

    def _handle_escape_key_pressed(self):
        if self.chat_manager and self.chat_manager.is_overall_busy():
            if hasattr(self.chat_manager, '_current_llm_request_id') and self.chat_manager._current_llm_request_id:
                request_to_cancel = self.chat_manager._current_llm_request_id
                if self.chat_manager._backend_coordinator:
                    self.chat_manager._backend_coordinator.cancel_current_task(request_id=request_to_cancel)
                    self.update_status("Attempting to cancel AI response...", "#e5c07b", True, 2000)

    def update_window_title(self):
        base_title = constants.APP_NAME
        details = []
        current_project: Optional[Project] = None
        current_session: Optional[ChatSession] = None

        if self._project_manager:
            current_project = self._project_manager.get_current_project()
            current_session = self._project_manager.get_current_session()

        if current_project:
            details.append(f"Project: {current_project.name[:25]}")
        if current_session:
            details.append(f"Session: {current_session.name[:25]}")

        if self.chat_manager:
            active_backend_id = self.chat_manager.get_current_active_chat_backend_id()
            model_name = self.chat_manager.get_model_for_backend(active_backend_id)
            if model_name:
                model_short = model_name.split('/')[-1].split(':')[-1].replace("-latest", "").replace("-preview-05-20","")
                details.append(f"LLM: {model_short[:20]}") # Shorten model name if too long
            if self.chat_manager.get_current_chat_personality():
                details.append("Persona")

        self.setWindowTitle(f"{base_title} - [{', '.join(details)}]" if details else base_title)


    def closeEvent(self, event: QCloseEvent):
        logger.info("MainWindow closeEvent triggered. Performing cleanup...")
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
            QTimer.singleShot(100, self.active_chat_input_bar.set_focus)
        QTimer.singleShot(150, self._clear_temporary_status)
        self.update_window_title()