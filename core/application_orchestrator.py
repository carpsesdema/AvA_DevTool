# core/application_orchestrator.py
import logging
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from PySide6.QtCore import QObject, Slot

try:
    from backends.backend_interface import BackendInterface
    from backends.gemini_adapter import GeminiAdapter
    from backends.ollama_adapter import OllamaAdapter
    from backends.gpt_adapter import GPTAdapter
    from backends.backend_coordinator import BackendCoordinator
    from core.event_bus import EventBus
    from services.llm_communication_logger import LlmCommunicationLogger
    from services.upload_service import UploadService
    from services.terminal_service import TerminalService
    from core.rag_handler import RagHandler
    from utils import constants
    from services.project_service import ProjectManager, Project, ChatSession
    from core.models import ChatMessage
    from ui.dialogs.code_viewer_dialog import CodeViewerWindow  # NEW
except ImportError as e:
    ProjectManager = type("ProjectManager", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    UploadService = type("UploadService", (object,), {})  # type: ignore
    RagHandler = type("RagHandler", (object,), {})  # type: ignore
    TerminalService = type("TerminalService", (object,), {})  # type: ignore
    CodeViewerWindow = type("CodeViewerWindow", (object,), {})  # type: ignore
    logging.getLogger(__name__).critical(f"Critical import error in ApplicationOrchestrator: {e}", exc_info=True)
    raise

# Conditional import to break circular dependency for type hinting
if TYPE_CHECKING:
    from core.chat_manager import ChatManager

logger = logging.getLogger(__name__)


class ApplicationOrchestrator(QObject):
    def __init__(self,
                 project_manager: ProjectManager,  # type: ignore
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ApplicationOrchestrator initializing...")

        self.event_bus = EventBus.get_instance()
        if self.event_bus is None:
            logger.critical("EventBus instance is None in ApplicationOrchestrator.")
            raise RuntimeError("EventBus could not be instantiated.")

        self.project_manager: ProjectManager = project_manager  # type: ignore
        if not isinstance(self.project_manager, ProjectManager):  # type: ignore
            logger.critical("ApplicationOrchestrator requires a valid ProjectManager.")
            raise TypeError("ApplicationOrchestrator requires a valid ProjectManager.")

        self.chat_manager: Optional = None  # Removed type hint to avoid circular import

        # --- LLM Backend Adapters ---
        self.gemini_chat_adapter = GeminiAdapter()
        self.ollama_chat_adapter = OllamaAdapter()
        self.gpt_chat_adapter = GPTAdapter()
        self.ollama_generator_adapter = OllamaAdapter()

        self._all_backend_adapters_dict: Dict[str, BackendInterface] = {
            "gemini_chat_default": self.gemini_chat_adapter,
            "ollama_chat_default": self.ollama_chat_adapter,
            "gpt_chat_default": self.gpt_chat_adapter,
            constants.GENERATOR_BACKEND_ID: self.ollama_generator_adapter
        }
        if constants.DEFAULT_CHAT_BACKEND_ID not in self._all_backend_adapters_dict:
            logger.warning(
                f"DEFAULT_CHAT_BACKEND_ID '{constants.DEFAULT_CHAT_BACKEND_ID}' not found in adapter map. Falling back to 'gemini_chat_default'.")
            self._all_backend_adapters_dict[constants.DEFAULT_CHAT_BACKEND_ID] = self.gemini_chat_adapter

        if constants.GENERATOR_BACKEND_ID not in self._all_backend_adapters_dict:
            logger.warning(
                f"GENERATOR_BACKEND_ID '{constants.GENERATOR_BACKEND_ID}' not found in adapter map. Attempting to add default Ollama Generator.")
            self._all_backend_adapters_dict[constants.GENERATOR_BACKEND_ID] = self.ollama_generator_adapter

        try:
            self.backend_coordinator = BackendCoordinator(self._all_backend_adapters_dict, parent=self)
        except ValueError as ve:
            logger.critical(f"Failed to instantiate BackendCoordinator: {ve}", exc_info=True)
            raise
        except Exception as e_bc:
            logger.critical(f"An unexpected error occurred instantiating BackendCoordinator: {e_bc}", exc_info=True)
            raise

        # --- RAG Services ---
        self.upload_service = None
        self.rag_handler = None

        try:
            self.upload_service = UploadService()
            vector_db_service = getattr(self.upload_service, '_vector_db_service',
                                        None) if self.upload_service else None
            self.rag_handler = RagHandler(self.upload_service, vector_db_service)
            logger.info("RAG services initialized successfully")
        except Exception as e_rag:
            logger.error(f"Failed to initialize RAG services: {e_rag}. RAG functionality will be disabled.")
            self.upload_service = None
            self.rag_handler = None

        # --- Terminal Service ---
        self.terminal_service = None
        try:
            self.terminal_service = TerminalService(parent=self)
            logger.info("TerminalService initialized successfully")
        except Exception as e_terminal:
            logger.error(
                f"Failed to initialize TerminalService: {e_terminal}. Terminal functionality will be disabled.")
            self.terminal_service = None

        # --- Code Viewer Dialog ---
        self.code_viewer_window = None
        try:
            self.code_viewer_window = CodeViewerWindow(parent=None)  # No parent so it can be independent
            logger.info("CodeViewerWindow initialized successfully")
        except Exception as e_code_viewer:
            logger.error(f"Failed to initialize CodeViewerWindow: {e_code_viewer}. Code viewer will be disabled.")
            self.code_viewer_window = None

        # --- LLM Communication Logger ---
        self.llm_communication_logger: Optional[LlmCommunicationLogger] = None
        try:
            self.llm_communication_logger = LlmCommunicationLogger(parent=self)
        except Exception as e_logger:
            logger.error(f"Failed to instantiate LlmCommunicationLogger: {e_logger}", exc_info=True)

        self._connect_event_bus()
        logger.info("ApplicationOrchestrator initialization complete.")

    def set_chat_manager(self, chat_manager):  # Removed type hint to avoid circular import
        """Allows ChatManager instance to be set after mutual creation."""
        self.chat_manager = chat_manager

    def _connect_event_bus(self):
        self.event_bus.createNewSessionForProjectRequested.connect(self._handle_create_new_session_requested)
        self.event_bus.createNewProjectRequested.connect(self._handle_create_new_project_requested)
        self.event_bus.messageFinalizedForSession.connect(self._handle_message_finalized_for_session_persistence)

        # Connect code viewer related signals
        self.event_bus.viewCodeViewerRequested.connect(self._handle_view_code_viewer_requested)
        self.event_bus.modificationFileReadyForDisplay.connect(self._handle_modification_file_ready_for_display)
        self.event_bus.applyFileChangeRequested.connect(self._handle_apply_file_change_requested)

        # Connect code viewer's apply signal to event bus
        if self.code_viewer_window:
            self.code_viewer_window.apply_change_requested.connect(
                lambda project_id, filepath, content, focus_prefix:
                self.event_bus.applyFileChangeRequested.emit(project_id, filepath, content, focus_prefix)
            )

        if hasattr(self.project_manager, 'projectDeleted'):
            self.project_manager.projectDeleted.connect(self._handle_project_deleted)  # type: ignore
        else:
            logger.warning("ProjectManager does not have projectDeleted signal. Cannot connect in Orchestrator.")

    @Slot()
    def _handle_view_code_viewer_requested(self):
        """Handle request to show the code viewer window"""
        if self.code_viewer_window:
            self.code_viewer_window.show()
            self.code_viewer_window.activateWindow()
            self.code_viewer_window.raise_()
            logger.info("Code viewer window shown")
        else:
            logger.error("Code viewer window not available")
            self.event_bus.uiErrorGlobal.emit("Code viewer not available", False)

    @Slot(str, str)
    def _handle_modification_file_ready_for_display(self, filename: str, content: str):
        """Handle when a file is ready to be displayed in the code viewer"""
        if not self.code_viewer_window:
            logger.error("Code viewer window not available for displaying file")
            return

        # Get current project context for apply functionality
        current_project_id = None
        if self.chat_manager:
            current_project_id = self.chat_manager.get_current_project_id()

        # Show the generated file in the code viewer
        self.code_viewer_window.update_or_add_file(
            filename=filename,
            content=content,
            is_ai_modification=True,
            original_content=None,  # For new files, no original content
            project_id_for_apply=current_project_id,
            focus_prefix_for_apply=self._get_current_project_directory()
        )

        logger.info(f"File '{filename}' displayed in code viewer")

    @Slot(str, str, str, str)
    def _handle_apply_file_change_requested(self, project_id: str, relative_filepath: str, new_content: str,
                                            focus_prefix: str):
        """Handle request to apply/save a file change to disk"""
        import os

        try:
            # Determine the full path for the file
            if focus_prefix and os.path.isabs(focus_prefix):
                full_path = os.path.join(focus_prefix, relative_filepath)
            else:
                # Use current working directory or project directory
                project_dir = self._get_current_project_directory()
                full_path = os.path.join(project_dir, relative_filepath)

            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            logger.info(f"Successfully applied file change: {full_path}")
            self.event_bus.uiStatusUpdateGlobal.emit(
                f"File saved: {os.path.basename(relative_filepath)}",
                "#98c379",
                True,
                3000
            )

            # Notify code viewer that apply completed
            if self.code_viewer_window:
                self.code_viewer_window.handle_apply_completed(relative_filepath)

        except Exception as e:
            logger.error(f"Error applying file change for {relative_filepath}: {e}")
            self.event_bus.uiErrorGlobal.emit(f"Failed to save file: {str(e)}", False)

    def _get_current_project_directory(self) -> str:
        """Get the current project's working directory"""
        # For now, return current working directory
        # This could be enhanced to use project-specific directories
        import os
        return os.getcwd()

    def initialize_application_state(self):
        logger.info("Orchestrator: Initializing application state (project/session)...")
        if not self.chat_manager:
            logger.error("Orchestrator: ChatManager not set. Cannot initialize application state.")
            return

        projects: List[Project] = self.project_manager.get_all_projects()  # type: ignore
        active_project: Optional[Project] = None  # type: ignore
        active_session: Optional[ChatSession] = None  # type: ignore

        if not projects:
            logger.info("Orchestrator: No projects found. Creating a default project.")
            active_project = self.project_manager.create_project(name="Default Project",
                                                                 description="My first project")  # type: ignore
            if active_project:
                self.project_manager.switch_to_project(active_project.id)  # type: ignore
                active_session = self.project_manager.get_current_session()  # type: ignore
        else:
            active_project = projects[0]
            self.project_manager.switch_to_project(active_project.id)  # type: ignore
            active_session = self.project_manager.get_current_session()  # type: ignore
            if not active_session and active_project:
                logger.info(f"Orchestrator: Project '{active_project.name}' has no active session. Creating one.")
                active_session = self.project_manager.create_session(active_project.id, "Main Chat")  # type: ignore
                if active_session:
                    self.project_manager.switch_to_session(active_session.id)  # type: ignore

        if active_project and active_session:
            logger.info(
                f"Orchestrator: Setting active session in ChatManager: P:{active_project.id}/S:{active_session.id}")
            history_to_set: List[ChatMessage] = active_session.message_history  # type: ignore
            self.chat_manager.set_active_session(active_project.id, active_session.id, history_to_set)
        else:
            logger.error(
                "Orchestrator: Failed to initialize or load a project/session. Chat functionality may be limited.")
            self.event_bus.uiErrorGlobal.emit("Failed to load or create an initial project/session.", True)

    @Slot(str, str)
    def _handle_create_new_project_requested(self, project_name: str, project_description: str):
        """Handle request to create a new project."""
        logger.info(f"Orchestrator: Request to create new project: '{project_name}'")

        if not project_name or not project_name.strip():
            logger.warning("Orchestrator: Cannot create project with empty name")
            self.event_bus.uiErrorGlobal.emit("Project name cannot be empty.", False)
            return

        try:
            new_project = self.project_manager.create_project(  # type: ignore
                name=project_name.strip(),
                description=project_description.strip() if project_description else ""
            )

            if new_project:
                logger.info(f"Orchestrator: Successfully created project '{new_project.name}' (ID: {new_project.id})")

                if self.project_manager.switch_to_project(new_project.id):  # type: ignore
                    current_session = self.project_manager.get_current_session()  # type: ignore

                    if current_session and self.chat_manager:
                        logger.info(
                            f"Orchestrator: Switching ChatManager to new project P:{new_project.id}/S:{current_session.id}")
                        history_to_set: List[ChatMessage] = current_session.message_history  # type: ignore
                        self.chat_manager.set_active_session(new_project.id, current_session.id, history_to_set)

                    self.event_bus.uiStatusUpdateGlobal.emit(f"Created and switched to project '{new_project.name}'",
                                                             "#98c379", True, 3000)
                else:
                    logger.error(f"Orchestrator: Created project but failed to switch to it: {new_project.id}")
                    self.event_bus.uiErrorGlobal.emit(f"Created project but failed to switch to it.", False)
            else:
                logger.error(f"Orchestrator: Failed to create project '{project_name}'")
                self.event_bus.uiErrorGlobal.emit(f"Failed to create project '{project_name}'.", False)

        except Exception as e:
            logger.error(f"Orchestrator: Error creating project '{project_name}': {e}", exc_info=True)
            self.event_bus.uiErrorGlobal.emit(f"Error creating project: {e}", False)

    @Slot(str)
    def _handle_create_new_session_requested(self, project_id: str):
        logger.info(f"Orchestrator: Request to create new session for project ID: {project_id}")
        if not self.chat_manager:
            logger.error("Orchestrator: ChatManager not available to handle new session.")
            return

        current_project = self.project_manager.get_project_by_id(project_id)  # type: ignore
        if not current_project:
            logger.error(f"Orchestrator: Project with ID {project_id} not found. Cannot create new session.")
            self.event_bus.uiErrorGlobal.emit(f"Project {project_id} not found.", False)
            return

        session_count = len(self.project_manager.get_project_sessions(project_id))  # type: ignore
        new_session_name = f"Chat Session {session_count + 1}"
        new_session = self.project_manager.create_session(project_id, new_session_name)  # type: ignore

        if new_session:
            self.project_manager.switch_to_session(new_session.id)  # type: ignore
            logger.info(
                f"Orchestrator: Switched to new session P:{project_id}/S:{new_session.id}. Updating ChatManager.")
            history_to_set: List[ChatMessage] = new_session.message_history  # type: ignore
            self.chat_manager.set_active_session(project_id, new_session.id, history_to_set)
        else:
            logger.error(f"Orchestrator: Failed to create new session for project {project_id}.")
            self.event_bus.uiErrorGlobal.emit(f"Failed to create new session in project {project_id}.", False)

    @Slot(str, str, str, ChatMessage, dict, bool)  # type: ignore
    def _handle_message_finalized_for_session_persistence(self,
                                                          project_id: str,
                                                          session_id: str,
                                                          request_id: str,
                                                          finalized_message_obj: ChatMessage,  # type: ignore
                                                          usage_stats_dict: dict,
                                                          is_error: bool):
        if not self.chat_manager:
            logger.error("Orchestrator: ChatManager not set. Cannot persist history.")
            return

        cm_pid = self.chat_manager.get_current_project_id()
        cm_sid = self.chat_manager.get_current_session_id()

        if project_id == cm_pid and session_id == cm_sid:
            logger.debug(
                f"Orchestrator: Persisting history for P:{project_id}/S:{session_id} after message finalization.")
            current_history = self.chat_manager.get_current_chat_history()
            self.project_manager.update_current_session_history(current_history)  # type: ignore
        else:
            logger.warning(f"Orchestrator: Received messageFinalized for P:{project_id}/S:{session_id}, "
                           f"but ChatManager active is P:{cm_pid}/S:{cm_sid}. History not persisted by this signal.")

    @Slot(str)
    def _handle_project_deleted(self, deleted_project_id: str):
        logger.info(f"Orchestrator: Project {deleted_project_id} was deleted.")
        if not self.chat_manager: return

        current_cm_pid = self.chat_manager.get_current_project_id()
        if current_cm_pid == deleted_project_id:
            logger.info(
                f"Orchestrator: Active project {deleted_project_id} was deleted. Initializing new default state.")
            self.initialize_application_state()

    def get_event_bus(self) -> EventBus:
        return self.event_bus

    def get_backend_coordinator(self) -> BackendCoordinator:
        if not hasattr(self, 'backend_coordinator') or self.backend_coordinator is None:
            logger.critical("BackendCoordinator accessed before proper initialization in Orchestrator.")
            raise AttributeError("BackendCoordinator not initialized.")
        return self.backend_coordinator

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:
        return self.llm_communication_logger

    def get_all_backend_adapters_dict(self) -> Dict[str, BackendInterface]:
        return self._all_backend_adapters_dict

    def get_project_manager(self) -> ProjectManager:  # type: ignore
        return self.project_manager

    def get_upload_service(self) -> UploadService:  # type: ignore
        """Returns the initialized UploadService instance."""
        return self.upload_service

    def get_rag_handler(self) -> RagHandler:  # type: ignore
        """Returns the initialized RagHandler instance."""
        return self.rag_handler

    def get_terminal_service(self) -> TerminalService:  # NEW GETTER
        """Returns the initialized TerminalService instance."""
        return self.terminal_service

    def get_code_viewer_window(self) -> Optional[CodeViewerWindow]:  # NEW GETTER
        """Returns the initialized CodeViewerWindow instance."""
        return self.code_viewer_window