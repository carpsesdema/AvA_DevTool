# core/application_orchestrator.py
import logging
from typing import Dict, Optional, Any, List, TYPE_CHECKING  # ADDED TYPE_CHECKING

from PySide6.QtCore import QObject, Slot

try:
    from backends.backend_interface import BackendInterface
    from backends.gemini_adapter import GeminiAdapter
    from backends.ollama_adapter import OllamaAdapter
    from backends.gpt_adapter import GPTAdapter
    from backends.backend_coordinator import BackendCoordinator
    from core.event_bus import EventBus
    from services.llm_communication_logger import LlmCommunicationLogger
    from utils import constants
    from services.project_service import ProjectManager, Project, ChatSession
    from core.models import ChatMessage
except ImportError as e:
    ProjectManager = type("ProjectManager", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    logging.getLogger(__name__).critical(f"Critical import error in ApplicationOrchestrator: {e}", exc_info=True)
    raise

# Conditional import to break circular dependency for type hinting
if TYPE_CHECKING:
    from core.chat_manager import ChatManager

logger = logging.getLogger(__name__)


class ApplicationOrchestrator(QObject):
    def __init__(self,
                 project_manager: ProjectManager,  # type: ignore
                 upload_service_placeholder: Any,
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

        self.chat_manager: Optional['ChatManager'] = None  # Use string literal for type hint

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

        self.llm_communication_logger: Optional[LlmCommunicationLogger] = None
        try:
            self.llm_communication_logger = LlmCommunicationLogger(parent=self)
        except Exception as e_logger:
            logger.error(f"Failed to instantiate LlmCommunicationLogger: {e_logger}", exc_info=True)

        self._upload_service_placeholder = upload_service_placeholder
        self._connect_event_bus()
        logger.info("ApplicationOrchestrator initialization complete.")

    def set_chat_manager(self, chat_manager: 'ChatManager'):  # Use string literal
        """Allows ChatManager instance to be set after mutual creation."""
        self.chat_manager = chat_manager

    def _connect_event_bus(self):
        self.event_bus.createNewSessionForProjectRequested.connect(self._handle_create_new_session_requested)
        self.event_bus.messageFinalizedForSession.connect(self._handle_message_finalized_for_session_persistence)
        if hasattr(self.project_manager, 'projectDeleted'):  # Check if signal exists before connecting
            self.project_manager.projectDeleted.connect(self._handle_project_deleted)  # type: ignore
        else:
            logger.warning("ProjectManager does not have projectDeleted signal. Cannot connect in Orchestrator.")

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
                if active_session:  # Check if session creation was successful
                    self.project_manager.switch_to_session(active_session.id)  # type: ignore

        if active_project and active_session:
            logger.info(
                f"Orchestrator: Setting active session in ChatManager: P:{active_project.id}/S:{active_session.id}")
            # Ensure ChatMessage type is correctly handled if it was conditionally imported
            history_to_set: List[ChatMessage] = active_session.message_history  # type: ignore
            self.chat_manager.set_active_session(active_project.id, active_session.id, history_to_set)
        else:
            logger.error(
                "Orchestrator: Failed to initialize or load a project/session. Chat functionality may be limited.")
            self.event_bus.uiErrorGlobal.emit("Failed to load or create an initial project/session.", True)

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
                                                          request_id: str,  # noqa
                                                          finalized_message_obj: ChatMessage,  # type: ignore # noqa
                                                          usage_stats_dict: dict,  # noqa
                                                          is_error: bool):  # noqa
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

    def get_upload_service_placeholder(self) -> Any:
        return self._upload_service_placeholder