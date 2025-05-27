# core/application_orchestrator.py
import logging
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from PySide6.QtCore import QObject, Slot


if TYPE_CHECKING:
    from core.chat_manager import ChatManager
    from backends.backend_interface import BackendInterface # Keep for type hinting if needed

logger = logging.getLogger(__name__)


class ApplicationOrchestrator(QObject):
    def __init__(self,
                 project_manager: 'ProjectManager', # Forward reference for ProjectManager
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ApplicationOrchestrator initializing...")

        # Moved imports inside __init__
        from backends.gemini_adapter import GeminiAdapter
        from backends.ollama_adapter import OllamaAdapter # This is the key import
        from backends.gpt_adapter import GPTAdapter
        from backends.backend_coordinator import BackendCoordinator
        from core.event_bus import EventBus
        from services.llm_communication_logger import LlmCommunicationLogger
        from services.upload_service import UploadService
        from services.terminal_service import TerminalService
        from services.update_service import UpdateService
        from core.rag_handler import RagHandler
        from utils import constants
        from services.project_service import ProjectManager, Project, ChatSession # ProjectManager now fully imported
        from core.models import ChatMessage
        # from ui.dialogs.code_viewer_dialog import CodeViewerWindow # Keep commented if not directly used by orchestrator

        self.event_bus = EventBus.get_instance()
        if self.event_bus is None:
            logger.critical("EventBus instance is None in ApplicationOrchestrator.")
            raise RuntimeError("EventBus could not be instantiated.")

        self.project_manager: ProjectManager = project_manager
        if not isinstance(self.project_manager, ProjectManager): # Now ProjectManager is fully imported
            logger.critical("ApplicationOrchestrator requires a valid ProjectManager.")
            raise TypeError("ApplicationOrchestrator requires a valid ProjectManager.")

        self.chat_manager: Optional['ChatManager'] = None

        self.gemini_chat_adapter = GeminiAdapter()
        self.ollama_chat_adapter = OllamaAdapter()
        self.gpt_chat_adapter = GPTAdapter()
        self.ollama_generator_adapter = OllamaAdapter() # Using another instance for generator

        self._all_backend_adapters_dict: Dict[str, 'BackendInterface'] = {
            "gemini_chat_default": self.gemini_chat_adapter,
            "ollama_chat_default": self.ollama_chat_adapter,
            "gpt_chat_default": self.gpt_chat_adapter,
            constants.GENERATOR_BACKEND_ID: self.ollama_generator_adapter
        }

        if constants.DEFAULT_CHAT_BACKEND_ID not in self._all_backend_adapters_dict:
            logger.critical(
                f"CRITICAL: constants.DEFAULT_CHAT_BACKEND_ID ('{constants.DEFAULT_CHAT_BACKEND_ID}') "
                f"is not a defined key in _all_backend_adapters_dict. This will cause issues. "
                f"Valid keys are: {list(self._all_backend_adapters_dict.keys())}. "
                f"Please ensure constants.DEFAULT_CHAT_BACKEND_ID matches one of these (e.g., 'gemini_chat_default')."
            )
            if "gemini_chat_default" in self._all_backend_adapters_dict:
                self._all_backend_adapters_dict[constants.DEFAULT_CHAT_BACKEND_ID] = self.gemini_chat_adapter
                logger.warning(
                    f"Fall-back: Pointing specific ID '{constants.DEFAULT_CHAT_BACKEND_ID}' to gemini_chat_adapter instance. Review constants.py for alignment.")

        if constants.GENERATOR_BACKEND_ID not in self._all_backend_adapters_dict:
            logger.warning(
                f"GENERATOR_BACKEND_ID '{constants.GENERATOR_BACKEND_ID}' not found in adapter map. "
                f"Re-adding default Ollama Generator."
            )
            self._all_backend_adapters_dict[constants.GENERATOR_BACKEND_ID] = self.ollama_generator_adapter

        try:
            self.backend_coordinator = BackendCoordinator(self._all_backend_adapters_dict, parent=self)
        except ValueError as ve:
            logger.critical(f"Failed to instantiate BackendCoordinator: {ve}", exc_info=True)
            raise
        except Exception as e_bc:
            logger.critical(f"An unexpected error occurred instantiating BackendCoordinator: {e_bc}", exc_info=True)
            raise

        self.upload_service: Optional[UploadService] = None
        self.rag_handler: Optional[RagHandler] = None
        try:
            self.upload_service = UploadService()
            vector_db_service = getattr(self.upload_service, '_vector_db_service',
                                        None) if self.upload_service else None
            self.rag_handler = RagHandler(self.upload_service, vector_db_service)
            logger.info("RAG services initialized successfully")
        except Exception as e_rag:
            logger.error(f"Failed to initialize RAG services: {e_rag}. RAG functionality will be disabled.",
                         exc_info=True)
            self.upload_service = None
            self.rag_handler = None

        self.terminal_service: Optional[TerminalService] = None
        try:
            self.terminal_service = TerminalService(parent=self)
            logger.info("TerminalService initialized successfully")
        except Exception as e_terminal:
            logger.error(
                f"Failed to initialize TerminalService: {e_terminal}. Terminal functionality will be disabled.",
                exc_info=True)
            self.terminal_service = None

        self.update_service: Optional[UpdateService] = None
        try:
            self.update_service = UpdateService(parent=self)
            logger.info("UpdateService initialized successfully")
        except Exception as e_update:
            logger.error(f"Failed to initialize UpdateService: {e_update}. Update functionality will be disabled.",
                         exc_info=True)
            self.update_service = None

        self.llm_communication_logger: Optional[LlmCommunicationLogger] = None
        try:
            self.llm_communication_logger = LlmCommunicationLogger(parent=self)
        except Exception as e_logger:
            logger.error(f"Failed to instantiate LlmCommunicationLogger: {e_logger}", exc_info=True)

        self._connect_event_bus()
        logger.info("ApplicationOrchestrator initialization complete.")

    def set_chat_manager(self, chat_manager: 'ChatManager'):
        self.chat_manager = chat_manager

    def _connect_event_bus(self):
        self.event_bus.createNewSessionForProjectRequested.connect(self._handle_create_new_session_requested)
        self.event_bus.createNewProjectRequested.connect(self._handle_create_new_project_requested)
        self.event_bus.messageFinalizedForSession.connect(self._handle_message_finalized_for_session_persistence)
        self.event_bus.modificationFileReadyForDisplay.connect(self._log_file_ready_for_display)
        self.event_bus.applyFileChangeRequested.connect(self._handle_apply_file_change_requested)

        if self.update_service:
            self.event_bus.checkForUpdatesRequested.connect(self.update_service.check_for_updates)
            self.update_service.update_available.connect(self.event_bus.updateAvailable.emit)
            self.update_service.no_update_available.connect(self.event_bus.noUpdateAvailable.emit)
            self.update_service.update_check_failed.connect(self.event_bus.updateCheckFailed.emit)
            self.update_service.update_downloaded.connect(self.event_bus.updateDownloaded.emit)
            self.update_service.update_download_failed.connect(self.event_bus.updateDownloadFailed.emit)
            self.update_service.update_progress.connect(self.event_bus.updateProgress.emit)
            self.update_service.update_status.connect(self.event_bus.updateStatusChanged.emit)
            self.event_bus.updateDownloadRequested.connect(self.update_service.download_update)
            self.event_bus.updateInstallRequested.connect(self._handle_update_install)
            self.event_bus.applicationRestartRequested.connect(self.update_service.restart_application)

        if hasattr(self.project_manager, 'projectDeleted'):
            self.project_manager.projectDeleted.connect(self._handle_project_deleted)
        else:
            logger.warning("ProjectManager does not have projectDeleted signal. Cannot connect in Orchestrator.")

    @Slot(str)
    def _handle_update_install(self, file_path: str):
        if self.update_service:
            success = self.update_service.apply_update(file_path)
            if success:
                logger.info("Update applied successfully, requesting restart")
                self.event_bus.applicationRestartRequested.emit()
            else:
                logger.error("Failed to apply update")
                self.event_bus.uiErrorGlobal.emit("Failed to install update", False)

    @Slot(str, str)
    def _log_file_ready_for_display(self, filename: str, content: str):
        logger.info(
            f"Orchestrator: Emitted modificationFileReadyForDisplay for '{filename}' ({len(content)} chars). MainWindow should handle UI.")

    @Slot(str, str, str, str)
    def _handle_apply_file_change_requested(self, project_id: str, relative_filepath: str, new_content: str,
                                            focus_prefix: str):
        import os
        try:
            if focus_prefix and os.path.isabs(focus_prefix) and os.path.isdir(
                    os.path.dirname(os.path.join(focus_prefix, relative_filepath))):
                base_dir_for_file = focus_prefix
            else:
                base_dir_for_file = self._get_current_project_directory(project_id_context=project_id)

            full_path = os.path.join(base_dir_for_file, relative_filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            logger.info(f"ApplicationOrchestrator: Successfully applied file change: {full_path}")
            self.event_bus.uiStatusUpdateGlobal.emit(
                f"File saved: {os.path.basename(relative_filepath)}", "#98c379", True, 3000
            )
        except Exception as e:
            logger.error(f"ApplicationOrchestrator: Error applying file change for {relative_filepath}: {e}",
                         exc_info=True)
            self.event_bus.uiErrorGlobal.emit(f"Failed to save file: {str(e)}", False)

    def _get_current_project_directory(self, project_id_context: Optional[str] = None) -> str:
        import os
        target_project_id = project_id_context
        if not target_project_id and self.chat_manager:
            target_project_id = self.chat_manager.get_current_project_id()

        if target_project_id and self.project_manager:
            return self.project_manager.get_project_files_dir(target_project_id)

        default_base = os.path.join(os.getcwd(), "ava_generated_projects")
        default_project_dir_name = "default_project_output"
        fallback_dir = os.path.join(default_base, default_project_dir_name)
        os.makedirs(fallback_dir, exist_ok=True)
        logger.warning(
            f"Orchestrator._get_current_project_directory: No specific project context, using fallback: {fallback_dir}")
        return fallback_dir

    def initialize_application_state(self):
        logger.info("Orchestrator: Initializing application state (project/session)...")
        if not self.chat_manager:
            logger.error("Orchestrator: ChatManager not set. Cannot initialize application state.")
            return

        projects: List['Project'] = self.project_manager.get_all_projects()
        active_project: Optional['Project'] = None
        active_session: Optional['ChatSession'] = None

        if not projects:
            logger.info("Orchestrator: No projects found. Creating a default project.")
            active_project = self.project_manager.create_project(name="Default Project", description="My first project")
            if active_project:
                self.project_manager.switch_to_project(active_project.id)
                active_session = self.project_manager.get_current_session()
        else:
            active_project = self.project_manager.get_current_project()
            if not active_project:
                active_project = projects[0]
                self.project_manager.switch_to_project(active_project.id)
            else:
                self.project_manager.switch_to_project(active_project.id)

            active_session = self.project_manager.get_current_session()
            if not active_session and active_project:
                logger.info(
                    f"Orchestrator: Project '{active_project.name}' has no current session. Trying to load/create one.")
                project_sessions = self.project_manager.get_project_sessions(active_project.id)
                if project_sessions:
                    active_session = project_sessions[0]
                    self.project_manager.switch_to_session(active_session.id)
                else:
                    active_session = self.project_manager.create_session(active_project.id, "Main Chat")
                    if active_session:
                        self.project_manager.switch_to_session(active_session.id)

        if active_project and active_session:
            logger.info(
                f"Orchestrator: Setting active session in ChatManager: P:{active_project.id}/S:{active_session.id}")
            history_to_set: List['ChatMessage'] = active_session.message_history
            self.chat_manager.set_active_session(active_project.id, active_session.id, history_to_set)
        else:
            logger.error(
                "Orchestrator: Failed to initialize or load a project/session. Chat functionality may be limited.")
            self.event_bus.uiErrorGlobal.emit("Failed to load or create an initial project/session.", True)

    @Slot(str, str)
    def _handle_create_new_project_requested(self, project_name: str, project_description: str):
        logger.info(f"Orchestrator: Request to create new project: '{project_name}'")
        if not project_name or not project_name.strip():
            logger.warning("Orchestrator: Cannot create project with empty name")
            self.event_bus.uiErrorGlobal.emit("Project name cannot be empty.", False)
            return
        try:
            new_project = self.project_manager.create_project(
                name=project_name.strip(),
                description=project_description.strip() if project_description else ""
            )
            if new_project:
                logger.info(f"Orchestrator: Successfully created project '{new_project.name}' (ID: {new_project.id})")
                if self.project_manager.switch_to_project(new_project.id):
                    current_session = self.project_manager.get_current_session()
                    if current_session and self.chat_manager:
                        logger.info(
                            f"Orchestrator: Switching ChatManager to new project P:{new_project.id}/S:{current_session.id}")
                        history_to_set: List['ChatMessage'] = current_session.message_history
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
        current_project = self.project_manager.get_project_by_id(project_id)
        if not current_project:
            logger.error(f"Orchestrator: Project with ID {project_id} not found. Cannot create new session.")
            self.event_bus.uiErrorGlobal.emit(f"Project {project_id} not found.", False)
            return
        session_count = len(self.project_manager.get_project_sessions(project_id))
        new_session_name = f"Chat Session {session_count + 1}"
        new_session = self.project_manager.create_session(project_id, new_session_name)
        if new_session:
            self.project_manager.switch_to_session(new_session.id)
            logger.info(
                f"Orchestrator: Switched to new session P:{project_id}/S:{new_session.id}. Updating ChatManager.")
            history_to_set: List['ChatMessage'] = new_session.message_history
            self.chat_manager.set_active_session(project_id, new_session.id, history_to_set)
        else:
            logger.error(f"Orchestrator: Failed to create new session for project {project_id}.")
            self.event_bus.uiErrorGlobal.emit(f"Failed to create new session in project {project_id}.", False)

    @Slot(str, str, str, object, dict, bool)
    def _handle_message_finalized_for_session_persistence(self,
                                                          project_id: str,
                                                          session_id: str,
                                                          request_id: str,
                                                          finalized_message_obj: 'ChatMessage',
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
            self.project_manager.update_current_session_history(current_history)
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

    def get_event_bus(self) -> 'EventBus':
        return self.event_bus

    def get_backend_coordinator(self) -> 'BackendCoordinator':
        if not hasattr(self, 'backend_coordinator') or self.backend_coordinator is None:
            logger.critical("BackendCoordinator accessed before proper initialization in Orchestrator.")
            raise AttributeError("BackendCoordinator not initialized.")
        return self.backend_coordinator

    def get_llm_communication_logger(self) -> Optional['LlmCommunicationLogger']:
        return self.llm_communication_logger

    def get_all_backend_adapters_dict(self) -> Dict[str, 'BackendInterface']:
        return self._all_backend_adapters_dict

    def get_project_manager(self) -> 'ProjectManager':
        return self.project_manager

    def get_upload_service(self) -> Optional['UploadService']:
        return self.upload_service

    def get_rag_handler(self) -> Optional['RagHandler']:
        return self.rag_handler

    def get_terminal_service(self) -> Optional['TerminalService']:
        return self.terminal_service

    def get_update_service(self) -> Optional['UpdateService']:
        return self.update_service