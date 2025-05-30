# core/application_orchestrator.py - Complete Enhanced Version
import logging
import os
from typing import Optional, Dict, Any

from PySide6.QtCore import QObject, Slot

# Forward references to handle potential circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.project_service import ProjectManager
    from core.chat_manager import ChatManager

logger = logging.getLogger(__name__)

try:
    from core.event_bus import EventBus
    from backends.backend_coordinator import BackendCoordinator
    from backends.gemini_adapter import GeminiAdapter
    from backends.ollama_adapter import OllamaAdapter
    from backends.gpt_adapter import GPTAdapter
    from services.upload_service import UploadService
    from services.terminal_service import TerminalService
    from services.update_service import UpdateService
    from services.llm_communication_logger import LlmCommunicationLogger
    from utils import constants

    # Import RAG handler with fallback
    try:
        from core.rag_handler import RagHandler
    except ImportError:
        logger.warning("RagHandler not found. RAG functionality will be limited.")
        RagHandler = None

    # Import RAG sync service
    try:
        from services.rag_sync_service import RagSyncService
    except ImportError:
        logger.warning("RagSyncService not available. Multi-project RAG sync disabled.")
        RagSyncService = None

except ImportError as e:
    logger.critical(f"Critical import error in ApplicationOrchestrator: {e}", exc_info=True)
    raise


class ApplicationOrchestrator(QObject):
    """Enhanced orchestrator with multi-project IDE support and RAG synchronization"""

    def __init__(self, project_manager: 'ProjectManager', parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ApplicationOrchestrator initializing with enhanced multi-project support...")

        # Core components
        self.event_bus = EventBus.get_instance()
        self.project_manager = project_manager
        self.chat_manager: Optional['ChatManager'] = None  # Set via set_chat_manager

        # LLM Backend Adapters
        self.gemini_adapter = GeminiAdapter()
        self.ollama_adapter = OllamaAdapter()
        self.gpt_adapter = GPTAdapter()
        self.ollama_generator_adapter = OllamaAdapter()  # Separate instance for generation

        # Backend mapping
        self._all_backend_adapters_dict = {
            constants.DEFAULT_CHAT_BACKEND_ID: self.gemini_adapter,
            "ollama_chat_default": self.ollama_adapter,
            "gpt_chat_default": self.gpt_adapter,
            constants.GENERATOR_BACKEND_ID: self.ollama_generator_adapter,
        }

        # Backend coordinator
        self.backend_coordinator = BackendCoordinator(
            backend_adapters=self._all_backend_adapters_dict,
            parent=self
        )

        # Initialize services
        self._initialize_core_services()

        # Connect event bus
        self._connect_event_bus()

        logger.info("ApplicationOrchestrator initialization complete with enhanced features.")

    def _initialize_core_services(self):
        """Initialize all core services including new RAG sync service"""

        # Upload service (RAG)
        try:
            self.upload_service = UploadService()
            logger.info("UploadService initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize UploadService: {e}")
            self.upload_service = None

        # RAG handler
        try:
            vector_db_service = getattr(self.upload_service, '_vector_db_service', None)
            self.rag_handler = RagHandler(
                upload_service=self.upload_service,
                vector_db_service=vector_db_service
            )
            logger.info("RagHandler initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize RagHandler: {e}")
            self.rag_handler = None

        # NEW: RAG sync service for multi-project IDE
        try:
            self.rag_sync_service = RagSyncService(
                upload_service=self.upload_service,
                project_manager=self.project_manager,
                parent=self
            )
            logger.info("RagSyncService initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to initialize RagSyncService (import): {e}")
            self.rag_sync_service = None
        except Exception as e:
            logger.error(f"Error initializing RagSyncService: {e}")
            self.rag_sync_service = None

        # Terminal service
        try:
            self.terminal_service = TerminalService(parent=self)
            logger.info("TerminalService initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize TerminalService: {e}")
            self.terminal_service = None

        # Update service
        try:
            self.update_service = UpdateService(parent=self)
            logger.info("UpdateService initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize UpdateService: {e}")
            self.update_service = None

        # LLM communication logger
        try:
            self.llm_communication_logger = LlmCommunicationLogger(parent=self)
            logger.info("LlmCommunicationLogger initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize LlmCommunicationLogger: {e}")
            self.llm_communication_logger = None

    def set_chat_manager(self, chat_manager: 'ChatManager'):
        """Set the chat manager reference"""
        self.chat_manager = chat_manager
        logger.info("ChatManager reference set in ApplicationOrchestrator.")

    def _connect_event_bus(self):
        """Connect EventBus signals to handler methods"""
        logger.info("Connecting EventBus signals for enhanced orchestrator...")

        # Existing connections
        self.event_bus.createNewSessionForProjectRequested.connect(self._handle_create_new_session_requested)
        self.event_bus.createNewProjectRequested.connect(self._handle_create_new_project_requested)
        self.event_bus.messageFinalizedForSession.connect(self._handle_message_finalized_for_session_persistence)
        self.event_bus.modificationFileReadyForDisplay.connect(self._log_file_ready_for_display)
        self.event_bus.applyFileChangeRequested.connect(self._handle_apply_file_change_requested)

        # Update service connections
        if self.update_service:
            self.event_bus.checkForUpdatesRequested.connect(self.update_service.check_for_updates)
            self.event_bus.updateDownloadRequested.connect(self.update_service.download_update)
            self.event_bus.updateInstallRequested.connect(self._handle_update_install)

            self.update_service.update_available.connect(self.event_bus.updateAvailable.emit)
            self.update_service.no_update_available.connect(self.event_bus.noUpdateAvailable.emit)
            self.update_service.update_check_failed.connect(self.event_bus.updateCheckFailed.emit)
            self.update_service.update_downloaded.connect(self.event_bus.updateDownloaded.emit)
            self.update_service.update_download_failed.connect(self.event_bus.updateDownloadFailed.emit)
            self.update_service.update_progress.connect(self.event_bus.updateProgress.emit)
            self.update_service.update_status.connect(self.event_bus.updateStatusChanged.emit)

        # Project manager connections
        if self.project_manager:
            self.project_manager.projectDeleted.connect(self._handle_project_deleted)

        # NEW: Multi-project IDE event connections
        self.event_bus.projectFilesSaved.connect(self._handle_project_file_saved)
        self.event_bus.projectLoaded.connect(self._handle_project_loaded_in_ide)
        self.event_bus.focusSetOnFiles.connect(self._handle_focus_set_on_files)
        self.event_bus.codeViewerProjectLoaded.connect(self._handle_code_viewer_project_loaded)

        logger.info("EventBus signals connected for enhanced orchestrator.")

    # Existing methods
    def _handle_update_install(self, file_path: str):
        """Handle update installation request"""
        if self.update_service:
            success = self.update_service.apply_update(file_path)
            if success:
                self.event_bus.applicationRestartRequested.emit()

    def _log_file_ready_for_display(self, filename: str, content: str):
        """Log that a file is ready for display"""
        logger.info(f"File ready for display: {filename}")

    def _handle_apply_file_change_requested(self, project_id: str, relative_filepath: str, new_content: str,
                                            focus_prefix: str):
        """Enhanced file change handler with project awareness"""
        try:
            # Determine the correct base directory
            if focus_prefix and os.path.isdir(focus_prefix):
                base_dir = focus_prefix
            else:
                # Try to get project directory from project manager
                if project_id and self.project_manager:
                    project = self.project_manager.get_project_by_id(project_id)
                    if project:
                        base_dir = self.project_manager.get_project_files_dir(project_id)
                    else:
                        base_dir = self._get_current_project_directory(project_id)
                else:
                    base_dir = self._get_current_project_directory(project_id)

            # Ensure directory exists
            os.makedirs(base_dir, exist_ok=True)

            # Write file
            full_path = os.path.join(base_dir, relative_filepath)
            file_dir = os.path.dirname(full_path)
            if file_dir:
                os.makedirs(file_dir, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            logger.info(f"Applied file change: {full_path}")

            # Emit file saved signal for RAG sync if we have a project_id
            if project_id:
                self.event_bus.projectFilesSaved.emit(project_id, full_path, new_content)

            # Emit status update
            self.event_bus.uiStatusUpdateGlobal.emit(
                f"File saved: {relative_filepath}", "#4ade80", True, 3000
            )

        except Exception as e:
            logger.error(f"Error applying file change: {e}")
            self.event_bus.uiErrorGlobal.emit(f"Failed to save {relative_filepath}: {e}", False)

    def _get_current_project_directory(self, project_id_context: Optional[str] = None) -> str:
        """Enhanced project directory resolution"""
        if project_id_context and self.project_manager:
            # Try to get project-specific directory
            project = self.project_manager.get_project_by_id(project_id_context)
            if project:
                return self.project_manager.get_project_files_dir(project_id_context)

        # Try current project from project manager
        if self.project_manager:
            current_project = self.project_manager.get_current_project()
            if current_project:
                return self.project_manager.get_project_files_dir(current_project.id)

        # Fallback to default directory
        default_dir = os.path.join(os.getcwd(), "ava_generated_projects", "default_project")
        os.makedirs(default_dir, exist_ok=True)
        return default_dir

    def initialize_application_state(self):
        """Initialize the application state with project and session"""
        logger.info("Initializing application state...")

        if not self.chat_manager:
            logger.error("ChatManager not set. Cannot initialize application state.")
            return

        try:
            # Load all projects
            if self.project_manager:
                projects = self.project_manager.get_all_projects()

                if not projects:
                    # Create default project
                    logger.info("No projects found. Creating default project.")
                    default_project = self.project_manager.create_project(
                        name="Default Project",
                        description="Default project for AvA conversations"
                    )
                    active_project = default_project
                else:
                    # Use the first project or last active one
                    active_project = projects[0]

                # Switch to the active project
                self.project_manager.switch_to_project(active_project.id)

                # Load or create a session for this project
                sessions = self.project_manager.get_project_sessions(active_project.id)
                if sessions:
                    # Switch to the most recent session
                    active_session = sessions[-1]
                    self.project_manager.switch_to_session(active_session.id)
                else:
                    # Create a new session
                    active_session = self.project_manager.create_session(
                        active_project.id, "Main Chat"
                    )
                    self.project_manager.switch_to_session(active_session.id)

                # Set the active session in ChatManager
                self.chat_manager.set_active_session(
                    active_project.id,
                    active_session.id,
                    active_session.message_history
                )

                logger.info(
                    f"Application state initialized with project '{active_project.name}' and session '{active_session.name}'.")

        except Exception as e:
            logger.error(f"Error initializing application state: {e}", exc_info=True)
            # Try to create a minimal fallback state
            try:
                if self.project_manager:
                    fallback_project = self.project_manager.create_project("Fallback Project")
                    fallback_session = self.project_manager.create_session(fallback_project.id, "Fallback Session")
                    self.chat_manager.set_active_session(
                        fallback_project.id, fallback_session.id, []
                    )
                    logger.info("Created fallback project/session state.")
            except Exception as fallback_error:
                logger.critical(f"Failed to create fallback state: {fallback_error}")

    def _handle_create_new_project_requested(self, project_name: str, project_description: str):
        """Handle new project creation requests"""
        if not self.project_manager:
            logger.error("ProjectManager not available for project creation.")
            return

        try:
            new_project = self.project_manager.create_project(project_name, project_description)
            self.project_manager.switch_to_project(new_project.id)

            # Update ChatManager if available
            if self.chat_manager:
                current_session = self.project_manager.get_current_session()
                if current_session:
                    self.chat_manager.set_active_session(
                        new_project.id,
                        current_session.id,
                        current_session.message_history
                    )

            logger.info(f"Created and switched to new project: {project_name}")

        except Exception as e:
            logger.error(f"Error creating new project: {e}")

    def _handle_create_new_session_requested(self, project_id: str):
        """Handle new session creation requests"""
        if not self.project_manager:
            logger.error("ProjectManager not available for session creation.")
            return

        try:
            new_session = self.project_manager.create_session(project_id,
                                                              f"Chat Session {len(self.project_manager.get_project_sessions(project_id)) + 1}")
            self.project_manager.switch_to_session(new_session.id)

            # Update ChatManager if available
            if self.chat_manager:
                self.chat_manager.set_active_session(
                    project_id,
                    new_session.id,
                    new_session.message_history
                )

            logger.info(f"Created and switched to new session in project: {project_id}")

        except Exception as e:
            logger.error(f"Error creating new session: {e}")

    def _handle_message_finalized_for_session_persistence(self, project_id: str, session_id: str, request_id: str,
                                                          message_obj, usage_stats_dict: dict, is_error: bool):
        """Handle message finalization for persistence"""
        if not self.chat_manager or not self.project_manager:
            return

        # Check if this is for the current active session
        current_project_id = self.chat_manager.get_current_project_id()
        current_session_id = self.chat_manager.get_current_session_id()

        if project_id == current_project_id and session_id == current_session_id:
            # Update the session history in project manager
            current_history = self.chat_manager.get_current_chat_history()
            self.project_manager.update_current_session_history(current_history)
            logger.debug(f"Updated session history for persistence: P:{project_id}/S:{session_id}")

    def _handle_project_deleted(self, deleted_project_id: str):
        """Handle project deletion"""
        if not self.chat_manager:
            return

        current_project_id = self.chat_manager.get_current_project_id()
        if deleted_project_id == current_project_id:
            # The active project was deleted, reinitialize to a valid state
            logger.info(f"Active project {deleted_project_id} was deleted. Reinitializing application state.")
            self.initialize_application_state()

    # NEW: Multi-project IDE event handlers
    @Slot(str, str, str)
    def _handle_project_file_saved(self, project_id: str, file_path: str, content: str):
        """Handle file saves from the CodeViewer IDE"""
        logger.info(f"Project file saved: {project_id} - {os.path.basename(file_path)}")

        # The RagSyncService will handle the actual RAG synchronization
        # This is just for logging and potential additional orchestration

        # Could add additional logic here like:
        # - Updating project statistics
        # - Triggering backups
        # - Notifying other services

    @Slot(str, str)
    def _handle_project_loaded_in_ide(self, project_id: str, project_path: str):
        """Handle project loading in CodeViewer IDE"""
        logger.info(f"Project loaded in IDE: {project_id} at {project_path}")

        # Check if this project exists in ProjectManager
        existing_project = self.project_manager.get_project_by_id(project_id)

        if not existing_project:
            # Auto-create project in ProjectManager to link IDE with project system
            project_name = os.path.basename(project_path)
            try:
                new_project = self.project_manager.create_project(
                    name=project_name,
                    description=f"Auto-created from IDE: {project_path}"
                )
                logger.info(f"Auto-created project {new_project.id} for IDE")

                # Update the project ID mapping if needed
                # Note: In a full implementation, you might want to update the IDE
                # to use the actual project ID from ProjectManager

            except Exception as e:
                logger.error(f"Failed to auto-create project for IDE: {e}")

    @Slot(str, list)
    def _handle_focus_set_on_files(self, project_id: str, file_paths: list):
        """Handle focus being set on files from CodeViewer"""
        logger.info(f"Focus set on {len(file_paths)} files in project {project_id}")

        # Store focus information for RAG query enhancement
        # This could be stored in a focus manager service or passed to ChatManager

        if hasattr(self, 'chat_manager') and self.chat_manager:
            # Could add a method to ChatManager to handle focus updates
            # self.chat_manager.update_project_focus(project_id, file_paths)
            pass

        # Emit status update
        self.event_bus.uiStatusUpdateGlobal.emit(
            f"AI focus set on {len(file_paths)} files",
            "#61dafb", True, 3000
        )

    @Slot(str, str, str)
    def _handle_code_viewer_project_loaded(self, project_name: str, project_path: str, project_id: str):
        """Handle CodeViewer-specific project loading"""
        logger.info(f"CodeViewer loaded project: {project_name}")

        # Emit the general project loaded signal
        self.event_bus.projectLoaded.emit(project_id, project_path)

    # Getter methods
    def get_event_bus(self) -> EventBus:
        """Get the EventBus instance"""
        return self.event_bus

    def get_backend_coordinator(self) -> BackendCoordinator:
        """Get the BackendCoordinator instance"""
        return self.backend_coordinator

    def get_upload_service(self) -> Optional[UploadService]:
        """Get the UploadService instance"""
        return self.upload_service

    def get_rag_handler(self) -> Optional[RagHandler]:
        """Get the RagHandler instance"""
        return self.rag_handler

    def get_rag_sync_service(self) -> Optional[RagSyncService]:
        """Get the RAG sync service instance"""
        return getattr(self, 'rag_sync_service', None)

    def get_terminal_service(self) -> Optional[TerminalService]:
        """Get the TerminalService instance"""
        return self.terminal_service

    def get_update_service(self) -> Optional[UpdateService]:
        """Get the UpdateService instance"""
        return self.update_service

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:
        """Get the LlmCommunicationLogger instance"""
        return self.llm_communication_logger

    def get_project_manager(self) -> 'ProjectManager':
        """Get the ProjectManager instance"""
        return self.project_manager

    # NEW: Project management helper methods
    def get_active_project_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently active project"""
        if not self.project_manager:
            return None

        current_project = self.project_manager.get_current_project()
        current_session = self.project_manager.get_current_session()

        return {
            'project': current_project,
            'session': current_session,
            'project_id': current_project.id if current_project else None,
            'session_id': current_session.id if current_session else None,
            'project_path': self.project_manager.get_project_files_dir() if current_project else None
        }

    def sync_project_with_ide(self, project_id: str, project_path: str) -> Optional[str]:
        """Sync a project between the IDE and project management system"""
        try:
            # Check if project exists in ProjectManager
            existing_project = self.project_manager.get_project_by_id(project_id)

            if not existing_project:
                # Create new project
                project_name = os.path.basename(project_path)
                new_project = self.project_manager.create_project(
                    name=project_name,
                    description=f"Synced from IDE: {project_path}"
                )
                logger.info(f"Created project {new_project.id} for IDE sync")
                return new_project.id
            else:
                logger.info(f"Project {project_id} already exists in ProjectManager")
                return project_id

        except Exception as e:
            logger.error(f"Error syncing project with IDE: {e}")
            return None

    def request_project_rag_initialization(self, project_id: str, project_path: str):
        """Request RAG initialization for a project"""
        if self.event_bus:
            self.event_bus.ragProjectInitializationRequested.emit(project_id, project_path)
            logger.info(f"Requested RAG initialization for project: {project_id}")

    def get_project_sync_status(self, project_id: str) -> Dict[str, Any]:
        """Get synchronization status for a project"""
        status = {
            'project_id': project_id,
            'exists_in_pm': False,
            'rag_ready': False,
            'sync_pending': False
        }

        if self.project_manager:
            project = self.project_manager.get_project_by_id(project_id)
            status['exists_in_pm'] = project is not None

        if self.upload_service:
            status['rag_ready'] = self.upload_service.is_vector_db_ready(project_id)

        if self.rag_sync_service:
            sync_status = self.rag_sync_service.get_sync_status(project_id)
            status['sync_pending'] = sync_status.get('pending_files', 0) > 0

        return status