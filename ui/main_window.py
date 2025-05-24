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
    from core.message_enums import MessageLoadingState, ApplicationMode
    from core.models import ChatMessage, SYSTEM_ROLE, ERROR_ROLE, MODEL_ROLE, USER_ROLE
    from ui.left_panel import LeftControlPanel
    from ui.dialog_service import DialogService
    from ui.chat_display_area import ChatDisplayArea
    from ui.chat_input_bar import ChatInputBar
    from utils import constants
    from services.llm_communication_logger import LlmCommunicationLogger
    from core.chat_message_state_handler import ChatMessageStateHandler
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
        logger.info("MainWindow initializing (Phase 1)...")
        if not isinstance(chat_manager, ChatManager):
            logger.critical("MainWindow requires a valid ChatManager instance.")
            raise TypeError("MainWindow requires a valid ChatManager instance.")

        self.chat_manager = chat_manager
        self.app_base_path = app_base_path
        self._event_bus = EventBus.get_instance()

        self.left_panel: Optional[LeftControlPanel] = None
        self.status_label: Optional[QLabel] = None
        self._status_clear_timer: Optional[QTimer] = None
        self.dialog_service: Optional[DialogService] = None
        self.p1_chat_display_area: Optional[ChatDisplayArea] = None
        self.p1_chat_input_bar: Optional[ChatInputBar] = None
        self._llm_comm_logger_instance: Optional[LlmCommunicationLogger] = None
        self._chat_message_state_handler: Optional[ChatMessageStateHandler] = None

        if self.chat_manager:
            self._llm_comm_logger_instance = self.chat_manager.get_llm_communication_logger()

        self._current_base_status_text: str = "Status: Initializing..."
        self._current_base_status_color: str = "#abb2bf"

        try:
            self.dialog_service = DialogService(self, self.chat_manager, self._event_bus)
        except Exception as e_ds:
            logger.critical(f"Failed to initialize DialogService in MainWindow: {e_ds}", exc_info=True)
            QApplication.quit()
            return

        self._init_ui_phase1()
        self._apply_styles()
        self._connect_signals_and_event_bus_phase1()
        self._setup_window_properties()
        logger.info("MainWindow (Phase 1) initialized successfully.")

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
        self.update_window_title_phase1()

    def _init_ui_phase1(self):
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

        self.p1_chat_display_area = ChatDisplayArea(parent=right_panel_widget)
        if self.p1_chat_display_area.chat_item_delegate:
            self.p1_chat_display_area.chat_item_delegate.setView(self.p1_chat_display_area.chat_list_view)

        self.p1_chat_input_bar = ChatInputBar(parent=right_panel_widget)

        right_panel_layout.addWidget(self.p1_chat_display_area, 1)
        right_panel_layout.addWidget(self.p1_chat_input_bar)

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
        if self.p1_chat_display_area.get_model():
            self._chat_message_state_handler.register_model_for_project(
                "p1_chat_context", self.p1_chat_display_area.get_model()
            )

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

    def _connect_signals_and_event_bus_phase1(self):
        if not all([self.chat_manager, self.left_panel, self.dialog_service, self.p1_chat_display_area,
                    self.p1_chat_input_bar]):
            return

        bus = self._event_bus

        if self.p1_chat_input_bar:
            self.p1_chat_input_bar.sendMessageRequested.connect(
                lambda: bus.userMessageSubmitted.emit(
                    self.p1_chat_input_bar.get_text() if self.p1_chat_input_bar else "",
                    self.p1_chat_input_bar.get_attached_image_data() if hasattr(self.p1_chat_input_bar,
                                                                                'get_attached_image_data') else []
                )
            )

        if self.p1_chat_display_area:
            self.p1_chat_display_area.textCopied.connect(
                lambda text, color: bus.uiTextCopied.emit(text, color)
            )

        bus.uiStatusUpdateGlobal.connect(self.update_status_phase1)
        bus.uiErrorGlobal.connect(self._handle_error_event_phase1)
        bus.uiInputBarBusyStateChanged.connect(self._handle_input_bar_busy_state_change_phase1)
        bus.backendConfigurationChanged.connect(self._handle_backend_configuration_changed_event_phase1)

        # FIXED: Connect to the code file display signal
        bus.modificationFileReadyForDisplay.connect(self._handle_code_file_update_event)

        bus.newMessageAddedToHistory.connect(self.p1_chat_display_area.add_message_to_model)
        bus.activeSessionHistoryCleared.connect(self.p1_chat_display_area.clear_model_display)
        bus.llmStreamChunkReceived.connect(self.p1_chat_display_area.append_chunk_to_message_by_id)
        bus.llmResponseCompleted.connect(
            lambda request_id, message, stats: self.p1_chat_display_area.finalize_message_by_id(request_id, message,
                                                                                                False)
        )
        bus.llmResponseError.connect(
            lambda request_id, error_msg: self.p1_chat_display_area.finalize_message_by_id(request_id, None, True)
        )

        shortcut_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        shortcut_escape.activated.connect(self._handle_escape_key_pressed_phase1)

    @Slot(str, str)
    def _handle_code_file_update_event(self, filename: str, content: str):
        logger.info(f"MainWindow: Received code update for '{filename}' via EventBus.")
        if self.dialog_service:
            code_viewer = self.dialog_service.show_code_viewer(ensure_creation=True)
            if code_viewer:
                project_id = "p1_chat_context"
                focus_prefix = self.app_base_path

                # For Phase 1, use simple defaults
                code_viewer.update_or_add_file(
                    filename,
                    content,
                    is_ai_modification=True,  # Since this comes from PlanAndCodeCoordinator
                    original_content=None,    # No original content tracking in Phase 1
                    project_id_for_apply=project_id,
                    focus_prefix_for_apply=focus_prefix
                )
                logger.info(f"MainWindow: Successfully added '{filename}' to CodeViewer")
            else:
                logger.error("MainWindow: CodeViewerDialog instance could not be obtained/created.")
        else:
            logger.error("MainWindow: DialogService not available to show CodeViewer.")

    @Slot(str, str, bool, int)
    def update_status_phase1(self, message: str, color: str, is_temporary: bool = False, duration_ms: int = 3000):
        if self.status_label is None: return
        self._current_base_status_text = message
        self._current_base_status_color = color
        self._refresh_full_status_display_phase1()

        if self._status_clear_timer:
            self._status_clear_timer.stop()
            self._status_clear_timer.deleteLater()
            self._status_clear_timer = None

        if is_temporary:
            self._status_clear_timer = QTimer(self)
            self._status_clear_timer.setSingleShot(True)
            self._status_clear_timer.timeout.connect(self._clear_temporary_status_phase1)
            self._status_clear_timer.start(duration_ms)

    def _refresh_full_status_display_phase1(self):
        if not self.status_label: return
        final_status_text = self._current_base_status_text
        self.status_label.setText(final_status_text)
        self.status_label.setStyleSheet(f"QLabel#StatusLabel {{ color: {self._current_base_status_color}; }}")

    def _clear_temporary_status_phase1(self):
        if self._status_clear_timer:
            self._status_clear_timer.stop()
            self._status_clear_timer.deleteLater()
            self._status_clear_timer = None

        if self.chat_manager and self.chat_manager.is_api_ready():
            self.update_status_phase1(
                f"Ready. Using {self.chat_manager.get_model_for_backend(self.chat_manager.get_current_active_chat_backend_id())}",
                "#98c379", is_temporary=False)
        elif self.chat_manager:
            last_error = self.chat_manager._backend_coordinator.get_last_error_for_backend(
                self.chat_manager.get_current_active_chat_backend_id()) if self.chat_manager._backend_coordinator else "Unknown"
            self.update_status_phase1(
                f"Backend not configured: {last_error}", "#FFCC00", is_temporary=False)
        else:
            self.update_status_phase1("Ready", "#98c379", is_temporary=False)

    @Slot(str, bool)
    def _handle_error_event_phase1(self, error_message: str, is_critical: bool):
        self.update_status_phase1(f"Error: {error_message[:100]}...", "#FF6B6B", True, 7000)
        if is_critical:
            QMessageBox.critical(self, "Critical Application Error", error_message)

    @Slot(bool)
    def _handle_input_bar_busy_state_change_phase1(self, is_input_bar_busy: bool):
        if self.p1_chat_input_bar:
            self.p1_chat_input_bar.handle_busy_state(is_input_bar_busy)
        if self.left_panel:
            api_ready = self.chat_manager.is_api_ready() if self.chat_manager else False
            self.left_panel.set_enabled_state(enabled=api_ready, is_busy=is_input_bar_busy)

    @Slot(str, str, bool, list)
    def _handle_backend_configuration_changed_event_phase1(self, backend_id: str, model_name: str, is_configured: bool,
                                                           available_models: list):
        self.update_window_title_phase1()
        if self.left_panel and self.chat_manager and backend_id == self.chat_manager.get_current_active_chat_backend_id():
            self.left_panel.update_personality_tooltip(active=bool(self.chat_manager.get_current_chat_personality()))

        if is_configured:
            logger.info(f"MW: Backend '{backend_id}' configured. Status updated via ChatManager.")
        else:
            logger.warning(f"MW: Backend '{backend_id}' not configured. Status updated via ChatManager.")

    def _handle_escape_key_pressed_phase1(self):
        if self.chat_manager and self.chat_manager.is_overall_busy():
            if hasattr(self.chat_manager,
                       '_current_llm_request_id') and self.chat_manager._current_llm_request_id:
                request_to_cancel = self.chat_manager._current_llm_request_id
                if self.chat_manager._backend_coordinator:
                    self.chat_manager._backend_coordinator.cancel_current_task(
                        request_id=request_to_cancel)
                    self.update_status_phase1("Attempting to cancel AI response...", "#e5c07b", True, 2000)

    def update_window_title_phase1(self):
        base_title = constants.APP_NAME
        details = []
        if self.chat_manager:
            active_backend_id = self.chat_manager.get_current_active_chat_backend_id()
            model_name = self.chat_manager.get_model_for_backend(active_backend_id)
            if model_name:
                model_short = model_name.split('/')[-1].split(':')[-1].replace("-latest", "")
                details.append(f"LLM: {model_short}")
            if self.chat_manager.get_current_chat_personality():
                details.append("Persona")

            spec_backend_id = constants.GENERATOR_BACKEND_ID
            spec_model_name = self.chat_manager.get_model_for_backend(spec_backend_id)
            if spec_model_name:
                spec_model_short = spec_model_name.split(':')[-1].split('/')[-1].replace("-latest", "")
                details.append(f"Coder: {spec_model_short}")

        self.setWindowTitle(f"{base_title} - [{', '.join(details)}]" if details else base_title)

    def closeEvent(self, event: QCloseEvent):
        logger.info("MainWindow closeEvent triggered. Performing P1 cleanup...")
        if self.dialog_service and hasattr(self.dialog_service, 'close_non_modal_dialogs'):
            self.dialog_service.close_non_modal_dialogs()
        if self.chat_manager and hasattr(self.chat_manager, 'cleanup_phase1'):
            self.chat_manager.cleanup_phase1()
        event.accept()

    def showEvent(self, event: QEvent):
        super().showEvent(event)
        if self.p1_chat_input_bar:
            QTimer.singleShot(100, self.p1_chat_input_bar.set_focus)

        if self.chat_manager:
            if self.chat_manager.is_api_ready():
                self.update_status_phase1(
                    f"Ready. Using {self.chat_manager.get_model_for_backend(self.chat_manager.get_current_active_chat_backend_id())}",
                    "#98c379")
            else:
                last_error = self.chat_manager._backend_coordinator.get_last_error_for_backend(
                    self.chat_manager.get_current_active_chat_backend_id()) if self.chat_manager._backend_coordinator else "Unknown"
                self.update_status_phase1(
                    f"Backend not configured: {last_error}", "#FFCC00")