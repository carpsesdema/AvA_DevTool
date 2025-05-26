# ui/left_panel.py
import logging
import os
from typing import Optional, Dict, List, Any

from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy,
    QComboBox, QGroupBox, QListWidget, QListWidgetItem, QHBoxLayout,
    QInputDialog, QMessageBox, QFileDialog
)

try:
    import qtawesome as qta

    QTAWESOME_AVAILABLE = True
except ImportError:
    QTAWESOME_AVAILABLE = False
    qta = None
    logging.getLogger(__name__).warning("LeftControlPanel: qtawesome library not found. Icons will be limited.")

try:
    from utils import constants
    from core.event_bus import EventBus
    from core.chat_manager import ChatManager
    from services.project_service import Project, ChatSession  # For type hinting
    # Assuming DialogService might be accessed via parent, or an event is used.
    # For direct call, MainWindow needs to expose it.
    # from ui.dialog_service import DialogService # Not directly imported if accessed via parent
except ImportError as e_lp:
    logging.getLogger(__name__).critical(f"Critical import error in LeftPanel: {e_lp}", exc_info=True)
    Project = type("Project", (object,), {})  # type: ignore
    ChatSession = type("ChatSession", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


class LeftControlPanel(QWidget):
    MODEL_CONFIG_DATA_ROLE = Qt.ItemDataRole.UserRole + 2
    PROJECT_ID_ROLE = Qt.ItemDataRole.UserRole + 3
    SESSION_ID_ROLE = Qt.ItemDataRole.UserRole + 4

    SPECIALIZED_BACKEND_DETAILS = [
        {"id": constants.GENERATOR_BACKEND_ID, "name": "Generator (Ollama)"},
    ]

    def __init__(self, chat_manager: ChatManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LeftControlPanel")
        if not isinstance(chat_manager, ChatManager):
            logger.critical("LeftControlPanel requires a valid ChatManager instance.")
            raise TypeError("LeftControlPanel requires a valid ChatManager instance.")

        self.chat_manager = chat_manager
        self._project_manager = chat_manager.get_project_manager()
        self._event_bus = EventBus.get_instance()
        self._is_programmatic_model_change: bool = False
        self._is_programmatic_selection_change: bool = False

        self._init_widgets_phase1()
        self._init_project_session_widgets()
        self._init_rag_widgets()
        self._init_layout_phase1()
        self._connect_signals_phase1()
        self._connect_project_session_signals()
        self._connect_rag_signals()

        self._load_initial_model_settings_phase1()
        self.load_initial_projects_and_sessions()

        logger.info("LeftControlPanel initialized.")

    def _get_qta_icon(self, icon_name: str, color: str = "#00CFE8") -> QIcon:
        if QTAWESOME_AVAILABLE and qta:
            try:
                return qta.icon(icon_name, color=color)
            except Exception:
                pass
        return QIcon()

    def _init_widgets_phase1(self):
        self.button_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1)
        self.button_style_sheet = "QPushButton { text-align: left; padding: 6px 8px; }"
        self.button_icon_size = QSize(16, 16)

        self.llm_config_group = QGroupBox("LLM Configuration")
        self.actions_group = QGroupBox("Chat Actions")
        self.projects_group = QGroupBox("Projects & Sessions")
        self.rag_group = QGroupBox("Knowledge Base (RAG)")

        for group_box in [self.llm_config_group, self.actions_group, self.projects_group,
                          self.rag_group]:
            group_box.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1, QFont.Weight.Bold))

        self.chat_llm_label = QLabel("Chat LLM:")
        self.chat_llm_label.setFont(self.button_font)

        self.chat_llm_combo_box = QComboBox()
        self.chat_llm_combo_box.setFont(self.button_font)
        self.chat_llm_combo_box.setObjectName("ChatLlmComboBox")
        self.chat_llm_combo_box.setToolTip("Select the primary AI model for chat")
        self.chat_llm_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.specialized_llm_label = QLabel("Specialized LLM:")
        self.specialized_llm_label.setFont(self.button_font)

        self.specialized_llm_combo_box = QComboBox()
        self.specialized_llm_combo_box.setFont(self.button_font)
        self.specialized_llm_combo_box.setObjectName("SpecializedLlmComboBox")
        self.specialized_llm_combo_box.setToolTip("Select the AI model for specialized tasks (e.g., code generation)")
        self.specialized_llm_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.configure_ai_personality_button = QPushButton(" Configure Persona")
        self.configure_ai_personality_button.setFont(self.button_font)
        self.configure_ai_personality_button.setIcon(self._get_qta_icon('fa5s.user-cog', color="#DAA520"))
        self.configure_ai_personality_button.setToolTip("Customize AI personality / system prompt (Ctrl+P)")
        self.configure_ai_personality_button.setObjectName("configureAiPersonalityButton")
        self.configure_ai_personality_button.setStyleSheet(self.button_style_sheet)
        self.configure_ai_personality_button.setIconSize(self.button_icon_size)

        self.new_chat_button = QPushButton(" New Session")
        self.new_chat_button.setFont(self.button_font)
        self.new_chat_button.setIcon(self._get_qta_icon('fa5s.comment-dots', color="#61AFEF"))
        self.new_chat_button.setToolTip("Start a new chat session in the current project (Ctrl+N)")
        self.new_chat_button.setObjectName("newChatButton")
        self.new_chat_button.setStyleSheet(self.button_style_sheet)
        self.new_chat_button.setIconSize(self.button_icon_size)

        self.view_llm_terminal_button = QPushButton(" View LLM Log")
        self.view_llm_terminal_button.setFont(self.button_font)
        self.view_llm_terminal_button.setIcon(self._get_qta_icon('fa5s.terminal', color="#98C379"))
        self.view_llm_terminal_button.setToolTip("Show LLM communication log (Ctrl+L)")
        self.view_llm_terminal_button.setObjectName("viewLlmTerminalButton")
        self.view_llm_terminal_button.setStyleSheet(self.button_style_sheet)
        self.view_llm_terminal_button.setIconSize(self.button_icon_size)

        self.view_generated_code_button = QPushButton(" View Generated Code")
        self.view_generated_code_button.setFont(self.button_font)
        self.view_generated_code_button.setIcon(self._get_qta_icon('fa5s.code', color="#ABB2BF"))
        self.view_generated_code_button.setToolTip("Open or focus the generated code viewer window")
        self.view_generated_code_button.setObjectName("viewGeneratedCodeButton")
        self.view_generated_code_button.setStyleSheet(self.button_style_sheet)
        self.view_generated_code_button.setIconSize(self.button_icon_size)

        # NEW: Add the update button
        self.check_updates_button = QPushButton(" Check for Updates")
        self.check_updates_button.setFont(self.button_font)
        self.check_updates_button.setIcon(self._get_qta_icon('fa5s.download', color="#E5C07B"))
        self.check_updates_button.setToolTip("Check for application updates")
        self.check_updates_button.setObjectName("checkUpdatesButton")
        self.check_updates_button.setStyleSheet(self.button_style_sheet)
        self.check_updates_button.setIconSize(self.button_icon_size)

    def _init_project_session_widgets(self):
        self.projects_list_widget = QListWidget()
        self.projects_list_widget.setObjectName("ProjectsListWidget")
        self.projects_list_widget.setFont(self.button_font)
        self.projects_list_widget.setToolTip("Select a project")
        self.projects_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.projects_list_widget.setMaximumHeight(150)

        self.sessions_list_widget = QListWidget()
        self.sessions_list_widget.setObjectName("SessionsListWidget")
        self.sessions_list_widget.setFont(self.button_font)
        self.sessions_list_widget.setToolTip("Select a session within the current project")
        self.sessions_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.new_project_button = QPushButton(" New Project")
        self.new_project_button.setFont(self.button_font)
        self.new_project_button.setIcon(self._get_qta_icon('fa5s.folder-plus', color="#56B6C2"))
        self.new_project_button.setToolTip("Create a new project")
        self.new_project_button.setObjectName("newProjectButton")
        self.new_project_button.setStyleSheet(self.button_style_sheet)
        self.new_project_button.setIconSize(self.button_icon_size)

    def _init_rag_widgets(self):
        self.scan_global_rag_directory_button = QPushButton(" Scan Directory (Global RAG)")
        self.scan_global_rag_directory_button.setFont(self.button_font)
        self.scan_global_rag_directory_button.setIcon(self._get_qta_icon('fa5s.globe-americas', color="#E0B6FF"))
        self.scan_global_rag_directory_button.setToolTip(
            "Scan a directory to add its files to the GLOBAL knowledge base.")
        self.scan_global_rag_directory_button.setObjectName("scanGlobalRagDirectoryButton")
        self.scan_global_rag_directory_button.setStyleSheet(self.button_style_sheet)
        self.scan_global_rag_directory_button.setIconSize(self.button_icon_size)

        self.add_project_files_button = QPushButton(" Add Files (Project RAG)")
        self.add_project_files_button.setFont(self.button_font)
        self.add_project_files_button.setIcon(self._get_qta_icon('fa5s.file-medical', color="#61AFEF"))
        self.add_project_files_button.setToolTip("Add specific files to the current project's knowledge base.")
        self.add_project_files_button.setObjectName("addProjectFilesButton")
        self.add_project_files_button.setStyleSheet(self.button_style_sheet)
        self.add_project_files_button.setIconSize(self.button_icon_size)

        self.rag_status_label = QLabel("RAG Status: Initializing...")
        self.rag_status_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 2))
        self.rag_status_label.setObjectName("RagStatusLabel")
        self.rag_status_label.setStyleSheet("QLabel#RagStatusLabel { color: #888888; }")
        self.rag_status_label.setWordWrap(True)

    def _init_layout_phase1(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        project_session_layout = QVBoxLayout(self.projects_group)
        project_session_layout.setSpacing(6)
        project_session_layout.addWidget(QLabel("Projects:"))
        project_session_layout.addWidget(self.projects_list_widget)
        project_session_layout.addWidget(self.new_project_button)
        project_session_layout.addSpacing(10)
        project_session_layout.addWidget(QLabel("Sessions (Current Project):"))
        project_session_layout.addWidget(self.sessions_list_widget)
        main_layout.addWidget(self.projects_group)

        llm_config_layout = QVBoxLayout(self.llm_config_group)
        llm_config_layout.setSpacing(6)
        llm_config_layout.addWidget(self.chat_llm_label)
        llm_config_layout.addWidget(self.chat_llm_combo_box)
        llm_config_layout.addWidget(self.specialized_llm_label)
        llm_config_layout.addWidget(self.specialized_llm_combo_box)
        llm_config_layout.addWidget(self.configure_ai_personality_button)
        main_layout.addWidget(self.llm_config_group)

        rag_actions_layout = QVBoxLayout(self.rag_group)
        rag_actions_layout.setSpacing(8)
        rag_actions_layout.addWidget(self.scan_global_rag_directory_button)
        rag_actions_layout.addWidget(self.add_project_files_button)
        rag_actions_layout.addWidget(self.rag_status_label)
        main_layout.addWidget(self.rag_group)

        actions_layout = QVBoxLayout(self.actions_group)
        actions_layout.setSpacing(6)
        actions_layout.addWidget(self.new_chat_button)
        actions_layout.addWidget(self.view_llm_terminal_button)
        actions_layout.addWidget(self.view_generated_code_button)
        actions_layout.addWidget(self.check_updates_button)  # NEW: Add the update button here
        main_layout.addWidget(self.actions_group)

        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _connect_signals_phase1(self):
        self.new_chat_button.clicked.connect(lambda: self._event_bus.newChatRequested.emit())
        self.configure_ai_personality_button.clicked.connect(
            lambda: self._event_bus.chatLlmPersonalityEditRequested.emit())
        self.view_llm_terminal_button.clicked.connect(lambda: self._event_bus.showLlmLogWindowRequested.emit())
        self.chat_llm_combo_box.currentIndexChanged.connect(self._on_chat_llm_selected_phase1)
        self.specialized_llm_combo_box.currentIndexChanged.connect(self._on_specialized_llm_selected_phase1)

        self._event_bus.backendConfigurationChanged.connect(self._handle_backend_configuration_changed_event_phase1)
        self._event_bus.backendBusyStateChanged.connect(self._handle_backend_busy_state_changed_event_phase1)
        self.view_generated_code_button.clicked.connect(lambda: self._event_bus.viewCodeViewerRequested.emit())

        # NEW: Connect the update button
        self.check_updates_button.clicked.connect(lambda: self._event_bus.checkForUpdatesRequested.emit())

    def _connect_project_session_signals(self):
        self.new_project_button.clicked.connect(self._on_new_project_requested)
        self.projects_list_widget.currentItemChanged.connect(self._on_project_selected)
        self.sessions_list_widget.currentItemChanged.connect(self._on_session_selected)

        if self._project_manager:
            self._project_manager.projectsLoaded.connect(self.populate_projects_list)
            self._project_manager.projectCreated.connect(self._handle_project_created)
            self._project_manager.sessionCreated.connect(self._handle_session_created)
            self._project_manager.projectSwitched.connect(self._handle_project_switched)
            self._project_manager.sessionSwitched.connect(self._handle_session_switched)

    def _connect_rag_signals(self):
        self.scan_global_rag_directory_button.clicked.connect(self._on_scan_global_rag_directory_requested)
        self.add_project_files_button.clicked.connect(self._on_add_project_files_requested)
        self._event_bus.ragStatusChanged.connect(self._handle_rag_status_changed)

    def load_initial_projects_and_sessions(self):
        logger.debug("LCP: Loading initial projects and sessions.")
        all_projects = self._project_manager.get_all_projects()
        self.populate_projects_list(all_projects)

        current_project = self._project_manager.get_current_project()
        if current_project:
            self._select_project_in_list(current_project.id)
            self.update_sessions_list(current_project.id)
        else:
            self.sessions_list_widget.clear()
        self.set_enabled_state(
            is_api_ready=self.chat_manager.is_api_ready(),
            is_busy=self.chat_manager.is_overall_busy(),
            is_rag_ready=self.chat_manager.is_rag_ready()
        )

    @Slot()
    def _on_new_project_requested(self):
        project_name, ok = QInputDialog.getText(self, "New Project", "Enter project name:")
        if ok and project_name:
            self._event_bus.createNewProjectRequested.emit(project_name, "")
        else:
            logger.info("New project creation cancelled or empty name.")

    @Slot()
    def _on_scan_global_rag_directory_requested(self):
        logger.info("LCP: 'Scan Directory (Global RAG)' button clicked.")
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Global RAG",
                                                     os.path.expanduser("~"))
        if directory:
            logger.info(f"LCP: User selected directory for GLOBAL RAG scan: {directory}")
            self._event_bus.requestRagScanDirectory.emit(directory)  # Emits only dir_path
        else:
            logger.info("LCP: Global RAG directory scan selection cancelled.")

    # Updated method for ui/left_panel.py

    @Slot()
    def _on_add_project_files_requested(self):
        logger.info("LCP: 'Add Files (Project RAG)' button clicked.")

        # Check if we have an active project
        if not self.chat_manager.get_current_project_id():
            logger.warning("LCP: No active project for RAG file addition.")
            self._event_bus.uiErrorGlobal.emit("No active project selected for adding RAG files.", False)
            return

        # Use EventBus to request the dialog - this is more robust than direct calls
        logger.debug("LCP: Emitting showProjectRagDialogRequested event.")
        self._event_bus.showProjectRagDialogRequested.emit()

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_project_selected(self, current: QListWidgetItem, previous: Optional[QListWidgetItem]):
        if self._is_programmatic_selection_change or not current:
            return
        project_id = current.data(self.PROJECT_ID_ROLE)
        if project_id and (not previous or project_id != previous.data(self.PROJECT_ID_ROLE)):
            logger.info(f"LCP: Project selected by user: {project_id}")
            self._project_manager.switch_to_project(project_id)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_session_selected(self, current: QListWidgetItem, previous: Optional[QListWidgetItem]):
        if self._is_programmatic_selection_change or not current:
            return
        session_id = current.data(self.SESSION_ID_ROLE)
        current_project = self._project_manager.get_current_project()
        if session_id and current_project and (not previous or session_id != previous.data(self.SESSION_ID_ROLE)):
            logger.info(f"LCP: Session selected by user: {session_id} in project {current_project.id}")
            self._project_manager.switch_to_session(session_id)

    @Slot(list)
    def populate_projects_list(self, projects: List[Project]):
        logger.debug(f"LCP: Populating projects list with {len(projects)} projects.")
        self._is_programmatic_selection_change = True
        self.projects_list_widget.clear()
        for project in projects:
            item = QListWidgetItem(project.name)
            item.setData(self.PROJECT_ID_ROLE, project.id)
            item.setToolTip(project.description or project.name)
            self.projects_list_widget.addItem(item)
        self._is_programmatic_selection_change = False

        current_proj_obj = self._project_manager.get_current_project()
        if current_proj_obj:
            self._select_project_in_list(current_proj_obj.id)

    @Slot(str)
    def _handle_project_created(self, project_id: str):
        project = self._project_manager.get_project_by_id(project_id)
        if project:
            self._is_programmatic_selection_change = True
            item = QListWidgetItem(project.name)
            item.setData(self.PROJECT_ID_ROLE, project.id)
            item.setToolTip(project.description or project.name)
            self.projects_list_widget.addItem(item)
            self._is_programmatic_selection_change = False
            self.projects_list_widget.setCurrentItem(item)

    @Slot(str)
    def _handle_project_switched(self, project_id: str):
        logger.debug(f"LCP: Handling project switched to {project_id}")
        self._select_project_in_list(project_id)
        self.update_sessions_list(project_id)
        self.set_enabled_state(
            is_api_ready=self.chat_manager.is_api_ready(),
            is_busy=self.chat_manager.is_overall_busy(),
            is_rag_ready=self.chat_manager.is_rag_ready()
        )

    @Slot(str, str)
    def _handle_session_created(self, project_id: str, session_id: str):
        current_project = self._project_manager.get_current_project()
        if current_project and current_project.id == project_id:
            session = self._project_manager.get_session_by_id(session_id)
            if session:
                self._is_programmatic_selection_change = True
                item = QListWidgetItem(session.name)
                item.setData(self.SESSION_ID_ROLE, session.id)
                self.sessions_list_widget.addItem(item)
                self._is_programmatic_selection_change = False
                self.sessions_list_widget.setCurrentItem(item)

    @Slot(str, str)
    def _handle_session_switched(self, project_id: str, session_id: str):
        logger.debug(f"LCP: Handling session switched to P:{project_id} S:{session_id}")
        current_project = self._project_manager.get_current_project()
        if current_project and current_project.id == project_id:
            self._select_session_in_list(session_id)
        self.set_enabled_state(
            is_api_ready=self.chat_manager.is_api_ready(),
            is_busy=self.chat_manager.is_overall_busy(),
            is_rag_ready=self.chat_manager.is_rag_ready()
        )

    def _select_project_in_list(self, project_id_to_select: str):
        self._is_programmatic_selection_change = True
        for i in range(self.projects_list_widget.count()):
            item = self.projects_list_widget.item(i)
            if item and item.data(self.PROJECT_ID_ROLE) == project_id_to_select:
                if self.projects_list_widget.currentItem() != item:
                    self.projects_list_widget.setCurrentItem(item)
                break
        self._is_programmatic_selection_change = False

    def _select_session_in_list(self, session_id_to_select: str):
        self._is_programmatic_selection_change = True
        for i in range(self.sessions_list_widget.count()):
            item = self.sessions_list_widget.item(i)
            if item and item.data(self.SESSION_ID_ROLE) == session_id_to_select:
                if self.sessions_list_widget.currentItem() != item:
                    self.sessions_list_widget.setCurrentItem(item)
                break
        self._is_programmatic_selection_change = False

    def update_sessions_list(self, project_id: str):
        logger.debug(f"LCP: Updating sessions list for project ID: {project_id}")
        self._is_programmatic_selection_change = True
        self.sessions_list_widget.clear()
        sessions = self._project_manager.get_project_sessions(project_id)
        for session in sessions:
            item = QListWidgetItem(session.name)
            item.setData(self.SESSION_ID_ROLE, session.id)
            self.sessions_list_widget.addItem(item)

        current_session_obj = self._project_manager.get_current_session()
        if current_session_obj and current_session_obj.project_id == project_id:
            self._select_session_in_list(current_session_obj.id)
        elif sessions:
            self._select_session_in_list(sessions[0].id)
            if not current_session_obj or current_session_obj.id != sessions[0].id:
                self._project_manager.switch_to_session(sessions[0].id)
        self._is_programmatic_selection_change = False

    def _load_initial_model_settings_phase1(self):
        self._is_programmatic_model_change = True
        self.chat_llm_combo_box.blockSignals(True)
        self.specialized_llm_combo_box.blockSignals(True)

        self._populate_chat_llm_combo_box_phase1()
        self._populate_specialized_llm_combo_box_phase1()

        active_chat_backend_id = self.chat_manager.get_current_active_chat_backend_id()
        active_chat_model_name = self.chat_manager.get_model_for_backend(active_chat_backend_id)
        self._set_combo_box_selection_phase1(self.chat_llm_combo_box, active_chat_backend_id, active_chat_model_name)

        active_specialized_backend_id = constants.GENERATOR_BACKEND_ID
        active_specialized_model_name = self.chat_manager.get_model_for_backend(active_specialized_backend_id)
        if not active_specialized_model_name:
            active_specialized_model_name = constants.DEFAULT_OLLAMA_GENERATOR_MODEL
            if self.chat_manager:
                self.chat_manager.set_model_for_backend(active_specialized_backend_id, active_specialized_model_name)

        self._set_combo_box_selection_phase1(self.specialized_llm_combo_box, active_specialized_backend_id,
                                             active_specialized_model_name)

        self.chat_llm_combo_box.blockSignals(False)
        self.specialized_llm_combo_box.blockSignals(False)
        self._is_programmatic_model_change = False
        self.update_personality_tooltip(active=bool(self.chat_manager.get_current_chat_personality()))
        self.chat_manager._check_rag_readiness_and_emit_status()
        self.set_enabled_state(
            is_api_ready=self.chat_manager.is_api_ready(),
            is_busy=self.chat_manager.is_overall_busy(),
            is_rag_ready=self.chat_manager.is_rag_ready()
        )

    def _populate_chat_llm_combo_box_phase1(self):
        self.chat_llm_combo_box.clear()
        models_added_count = 0
        all_backend_ids = self.chat_manager.get_all_available_backend_ids()
        user_selectable_chat_ids = {"gemini_chat_default", "ollama_chat_default", "gpt_chat_default"}
        for backend_id in all_backend_ids:
            if backend_id not in user_selectable_chat_ids: continue
            available_models_for_backend = self.chat_manager.get_available_models_for_backend(backend_id)
            if not available_models_for_backend:
                if backend_id == "gemini_chat_default":
                    available_models_for_backend = [constants.DEFAULT_GEMINI_CHAT_MODEL]
                elif backend_id == "ollama_chat_default":
                    available_models_for_backend = [constants.DEFAULT_OLLAMA_CHAT_MODEL]
                elif backend_id == "gpt_chat_default":
                    available_models_for_backend = ["gpt-4o", "gpt-3.5-turbo"]
            for model_name_str in available_models_for_backend:
                display_name_prefix = ""
                if backend_id == "gemini_chat_default":
                    display_name_prefix = "Gemini: "; model_name_display = model_name_str.replace("models/", "")
                elif backend_id == "ollama_chat_default":
                    display_name_prefix = "Ollama: "; model_name_display = model_name_str
                elif backend_id == "gpt_chat_default":
                    display_name_prefix = "GPT: "; model_name_display = model_name_str
                else:
                    model_name_display = model_name_str
                item_display_text = f"{display_name_prefix}{model_name_display}"
                user_data_for_item = {"backend_id": backend_id, "model_name": model_name_str}
                self.chat_llm_combo_box.addItem(item_display_text, userData=user_data_for_item)
                models_added_count += 1
        if models_added_count == 0:
            self.chat_llm_combo_box.addItem("No Chat LLMs Available"); self.chat_llm_combo_box.setEnabled(False)
        else:
            self.chat_llm_combo_box.setEnabled(True)

    def _populate_specialized_llm_combo_box_phase1(self):
        self.specialized_llm_combo_box.clear()
        models_added_count = 0
        for backend_detail in self.SPECIALIZED_BACKEND_DETAILS:
            backend_id = backend_detail["id"];
            backend_display_name = backend_detail["name"]
            available_models = self.chat_manager.get_available_models_for_backend(backend_id)
            if not available_models and backend_id == constants.GENERATOR_BACKEND_ID: available_models = [
                constants.DEFAULT_OLLAMA_GENERATOR_MODEL]
            for model_name_str in available_models:
                item_display_text = f"{backend_display_name}: {model_name_str}"
                user_data_for_item = {"backend_id": backend_id, "model_name": model_name_str}
                self.specialized_llm_combo_box.addItem(item_display_text, userData=user_data_for_item)
                models_added_count += 1
        if models_added_count == 0:
            self.specialized_llm_combo_box.addItem(
                "No Specialized LLMs Available"); self.specialized_llm_combo_box.setEnabled(False)
        else:
            self.specialized_llm_combo_box.setEnabled(True)

    def _set_combo_box_selection_phase1(self, combo_box: QComboBox, target_backend_id: str,
                                        target_model_name: Optional[str]):
        for i in range(combo_box.count()):
            item_data = combo_box.itemData(i)
            if isinstance(item_data, dict) and item_data.get("backend_id") == target_backend_id and item_data.get(
                    "model_name") == target_model_name:
                if combo_box.currentIndex() != i: combo_box.setCurrentIndex(i)
                return
        for i in range(combo_box.count()):
            item_data = combo_box.itemData(i)
            if isinstance(item_data, dict) and item_data.get("backend_id") == target_backend_id:
                if combo_box.currentIndex() != i: combo_box.setCurrentIndex(i)
                return
        if combo_box.count() > 0: combo_box.setCurrentIndex(0)

    @Slot(int)
    def _on_chat_llm_selected_phase1(self, index: int):
        if self._is_programmatic_model_change or index < 0: return
        selected_data = self.chat_llm_combo_box.itemData(index)
        if not isinstance(selected_data,
                          dict) or "backend_id" not in selected_data or "model_name" not in selected_data:
            logger.warning(f"LP: Invalid item data selected in chat LLM combo box: {selected_data}");
            return
        backend_id, model_name = selected_data["backend_id"], selected_data["model_name"]
        if self.chat_manager: logger.info(
            f"LP: User selected chat LLM. Backend: '{backend_id}', Model: '{model_name}'. Emitting to EventBus."); self._event_bus.chatLlmSelectionChanged.emit(
            backend_id, model_name)

    @Slot(int)
    def _on_specialized_llm_selected_phase1(self, index: int):
        if self._is_programmatic_model_change or index < 0: return
        selected_data = self.specialized_llm_combo_box.itemData(index)
        if not isinstance(selected_data,
                          dict) or "backend_id" not in selected_data or "model_name" not in selected_data:
            logger.warning(f"LP: Invalid item data selected in specialized LLM combo box: {selected_data}");
            return
        backend_id, model_name = selected_data["backend_id"], selected_data["model_name"]
        if self.chat_manager: logger.info(
            f"LP: User selected specialized LLM. Backend: '{backend_id}', Model: '{model_name}'. Emitting to EventBus."); self._event_bus.specializedLlmSelectionChanged.emit(
            backend_id, model_name)

    @Slot(str, str, bool, list)
    def _handle_backend_configuration_changed_event_phase1(self, backend_id: str, model_name: str, is_configured: bool,
                                                           available_models: list[Any]):
        logger.debug(f"LP: Backend config changed event for '{backend_id}'. Updating combo boxes.")
        self._is_programmatic_model_change = True
        self.chat_llm_combo_box.blockSignals(True);
        self.specialized_llm_combo_box.blockSignals(True)
        current_chat_backend = self.chat_manager.get_current_active_chat_backend_id();
        current_chat_model = self.chat_manager.get_model_for_backend(current_chat_backend)
        current_spec_backend = constants.GENERATOR_BACKEND_ID;
        current_spec_model = self.chat_manager.get_model_for_backend(current_spec_backend)
        self._populate_chat_llm_combo_box_phase1();
        self._populate_specialized_llm_combo_box_phase1()
        self._set_combo_box_selection_phase1(self.chat_llm_combo_box, current_chat_backend, current_chat_model)
        self._set_combo_box_selection_phase1(self.specialized_llm_combo_box, current_spec_backend, current_spec_model)
        self.chat_llm_combo_box.blockSignals(False);
        self.specialized_llm_combo_box.blockSignals(False)
        self._is_programmatic_model_change = False
        self.update_personality_tooltip(active=bool(self.chat_manager.get_current_chat_personality()))
        self.set_enabled_state(
            is_api_ready=self.chat_manager.is_api_ready(),
            is_busy=self.chat_manager.is_overall_busy(),
            is_rag_ready=self.chat_manager.is_rag_ready()
        )

    @Slot(bool, str, str)
    def _handle_rag_status_changed(self, is_ready: bool, status_text: str, status_color: str):
        logger.debug(f"LCP: RAG Status Changed: Ready={is_ready}, Text='{status_text}', Color='{status_color}'")
        self.rag_status_label.setText(f"{status_text}")
        self.rag_status_label.setStyleSheet(f"QLabel#RagStatusLabel {{ color: {status_color}; }}")
        self.set_enabled_state(
            is_api_ready=self.chat_manager.is_api_ready(),
            is_busy=self.chat_manager.is_overall_busy(),
            is_rag_ready=is_ready
        )

    @Slot(bool)
    def _handle_backend_busy_state_changed_event_phase1(self, is_busy: bool):
        self.set_enabled_state(
            is_api_ready=self.chat_manager.is_api_ready(),
            is_busy=is_busy,
            is_rag_ready=self.chat_manager.is_rag_ready()
        )

    def update_personality_tooltip(self, active: bool):
        tooltip_base = "Customize AI personality / system prompt (Ctrl+P)"
        status = "(Custom Persona Active)" if active else "(Default Persona)"
        self.configure_ai_personality_button.setToolTip(f"{tooltip_base}\nStatus: {status}")

    def set_enabled_state(self, is_api_ready: bool, is_busy: bool, is_rag_ready: bool):
        effective_enabled_not_busy = is_api_ready and not is_busy
        is_project_active = self.chat_manager.get_current_project_id() is not None

        self.chat_llm_combo_box.setEnabled(is_api_ready)
        self.specialized_llm_combo_box.setEnabled(is_api_ready)
        self.configure_ai_personality_button.setEnabled(effective_enabled_not_busy)

        self.new_chat_button.setEnabled(not is_busy and is_project_active)
        self.view_llm_terminal_button.setEnabled(True)
        self.view_generated_code_button.setEnabled(True)
        self.check_updates_button.setEnabled(not is_busy)  # NEW: Add the update button state

        self.projects_list_widget.setEnabled(not is_busy)
        self.sessions_list_widget.setEnabled(not is_busy and is_project_active)
        self.new_project_button.setEnabled(not is_busy)

        self.scan_global_rag_directory_button.setEnabled(not is_busy and is_rag_ready)
        self.add_project_files_button.setEnabled(not is_busy and is_rag_ready and is_project_active)

        label_color = "#C0C0C0" if is_api_ready else "#707070"
        self.chat_llm_label.setStyleSheet(f"QLabel {{ color: {label_color}; }}")
        self.specialized_llm_label.setStyleSheet(f"QLabel {{ color: {label_color}; }}")