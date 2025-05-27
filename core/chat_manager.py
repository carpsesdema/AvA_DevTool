# core/chat_manager.py - Key fixes for async RAG and auto LLM terminal
import asyncio
import logging
import os
import uuid
import re
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from PySide6.QtCore import QObject, Slot

# Move the TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from core.application_orchestrator import ApplicationOrchestrator

try:
    from core.event_bus import EventBus
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from core.message_enums import MessageLoadingState
    from backends.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from services.project_service import ProjectManager
    from services.upload_service import UploadService
    from core.rag_handler import RagHandler
    from utils import constants
    from config import get_gemini_api_key, get_openai_api_key
    from core.user_input_handler import UserInputHandler, UserInputIntent
    from core.plan_and_code_coordinator import PlanAndCodeCoordinator
except ImportError as e:
    ProjectManager = type("ProjectManager", (object,), {})
    UploadService = type("UploadService", (object,), {})
    RagHandler = type("RagHandler", (object,), {})
    logging.getLogger(__name__).critical(f"Critical import error in ChatManager: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatManager(QObject):
    def __init__(self, orchestrator, parent: Optional[QObject] = None):  # Removed type hint to avoid circular import
        super().__init__(parent)
        logger.info("ChatManager initializing...")

        self._orchestrator = orchestrator

        self._event_bus = self._orchestrator.get_event_bus()
        self._backend_coordinator = self._orchestrator.get_backend_coordinator()
        self._llm_comm_logger = self._orchestrator.get_llm_communication_logger()
        self._project_manager = self._orchestrator.get_project_manager()

        self._upload_service = self._orchestrator.get_upload_service()
        self._rag_handler = self._orchestrator.get_rag_handler()

        if not isinstance(self._project_manager, ProjectManager):
            logger.critical("ChatManager received an invalid ProjectManager instance.")
            raise TypeError("ChatManager requires a valid ProjectManager instance from Orchestrator.")

        self._user_input_handler = UserInputHandler()
        self._plan_and_code_coordinator = PlanAndCodeCoordinator(
            backend_coordinator=self._backend_coordinator,
            event_bus=self._event_bus,
            llm_comm_logger=self._llm_comm_logger,
            parent=self
        )

        self._current_chat_history: List[ChatMessage] = []
        self._current_project_id: Optional[str] = None
        self._current_session_id: Optional[str] = None
        self._active_chat_backend_id: str = constants.DEFAULT_CHAT_BACKEND_ID
        self._active_chat_model_name: str = (
            constants.DEFAULT_OLLAMA_CHAT_MODEL
            if constants.DEFAULT_CHAT_BACKEND_ID == "ollama_chat_default"
            else constants.DEFAULT_GEMINI_CHAT_MODEL
        )
        self._active_chat_personality_prompt: Optional[
            str] = "You are Ava, a bubbly, enthusiastic, and incredibly helpful AI assistant!"
        self._active_specialized_backend_id: str = constants.GENERATOR_BACKEND_ID
        self._active_specialized_model_name: str = constants.DEFAULT_OLLAMA_GENERATOR_MODEL
        self._active_temperature: float = 0.7
        self._is_chat_backend_configured: bool = False
        self._is_specialized_backend_configured: bool = False
        self._is_rag_ready: bool = False  # Represents if RAG for *current context* is generally usable
        self._current_llm_request_id: Optional[str] = None

        # NEW: File creation tracking
        self._file_creation_request_ids: Dict[str, str] = {}  # request_id -> filename

        # NEW: LLM terminal auto-open tracking
        self._llm_terminal_opened: bool = False

        self._connect_event_bus_subscriptions()
        logger.info("ChatManager initialized and subscriptions connected.")

    def initialize(self):
        """Delayed initialization to avoid blocking startup"""
        logger.info("ChatManager late initialization...")

        # NEW: Use timer to delay backend configuration to avoid blocking UI
        from PySide6.QtCore import QTimer

        def configure_backends():
            logger.info("ChatManager: Configuring backends asynchronously...")
            self._configure_backend(self._active_chat_backend_id, self._active_chat_model_name,
                                    self._active_chat_personality_prompt)
            self._configure_backend(self._active_specialized_backend_id, self._active_specialized_model_name,
                                    constants.CODER_AI_SYSTEM_PROMPT)
            self._check_rag_readiness_and_emit_status()

        # Configure backends after a short delay
        QTimer.singleShot(300, configure_backends)

    def set_active_session(self, project_id: str, session_id: str, history: List[ChatMessage]):
        logger.info(f"CM: Setting active session to P:{project_id}/S:{session_id}. History items: {len(history)}")
        old_project_id = self._current_project_id
        self._current_project_id = project_id
        self._current_session_id = session_id
        self._current_chat_history = list(history)

        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(self._current_llm_request_id)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._event_bus.hideLoader.emit()  # Ensure loader is hidden if a task was cancelled

        # Clear any pending file creation requests when switching sessions
        self._file_creation_request_ids.clear()

        self._event_bus.activeSessionHistoryCleared.emit(project_id, session_id)
        self._event_bus.activeSessionHistoryLoaded.emit(project_id, session_id, self._current_chat_history)
        self._emit_status_update(f"Switched to session.", "#98c379", True, 2000)

        # Always re-check RAG status on session switch, as project might have changed or RAG state updated
        self._check_rag_readiness_and_emit_status()

    def _connect_event_bus_subscriptions(self):
        logger.debug("ChatManager connecting EventBus subscriptions...")
        self._event_bus.userMessageSubmitted.connect(self.handle_user_message)
        self._event_bus.newChatRequested.connect(self.request_new_chat_session)
        self._event_bus.chatLlmPersonalitySubmitted.connect(self._handle_personality_change_event)
        self._event_bus.chatLlmSelectionChanged.connect(self._handle_chat_llm_selection_event)
        self._event_bus.specializedLlmSelectionChanged.connect(self._handle_specialized_llm_selection_event)
        self._event_bus.llmRequestSent.connect(self._handle_llm_request_sent)
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_response_completed)
        self._event_bus.llmResponseError.connect(self._handle_llm_response_error)
        self._event_bus.backendConfigurationChanged.connect(self._handle_backend_reconfigured_event)

        # MODIFICATION: Connect to the correct RAG signals
        self._event_bus.requestRagScanDirectory.connect(self.request_global_rag_scan_directory)  # For global
        self._event_bus.requestProjectFilesUpload.connect(
            self.handle_project_files_upload_request)  # For project-specific

        logger.debug("ChatManager EventBus subscriptions connected.")

    def _emit_status_update(self, message: str, color: str, is_temporary: bool = False, duration_ms: int = 0):
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, is_temporary, duration_ms)

    def _log_llm_comm(self, sender: str, message: str):
        if self._llm_comm_logger:
            log_prefix = f"[P:{self._current_project_id[:6]}/S:{self._current_session_id[:6]}]" if self._current_project_id and self._current_session_id else "[NoActiveSession]"
            self._llm_comm_logger.log_message(f"{log_prefix} {sender}", message)

            # NEW: Auto-open LLM terminal on first log message if not already opened
            if not self._llm_terminal_opened:
                self._llm_terminal_opened = True
                logger.info("ChatManager: Auto-opening LLM terminal for first communication")
                self._event_bus.showLlmLogWindowRequested.emit()
        else:
            logger.info(f"LLM_COMM_LOG_FALLBACK: [{sender}] {message[:150]}...")

    def _configure_backend(self, backend_id: str, model_name: str, system_prompt: Optional[str]):
        """Configure backend without blocking UI"""
        logger.info(f"CM: Configuring backend '{backend_id}' with model '{model_name}'")

        try:
            api_key_to_use: Optional[str] = None
            actual_system_prompt = system_prompt
            if backend_id == constants.GENERATOR_BACKEND_ID:
                actual_system_prompt = constants.CODER_AI_SYSTEM_PROMPT
                logger.info(f"CM: Using CODER_AI_SYSTEM_PROMPT for {backend_id}")
            if backend_id == "gemini_chat_default":
                api_key_to_use = get_gemini_api_key()
            elif backend_id == "gpt_chat_default":
                api_key_to_use = get_openai_api_key()
            elif backend_id in [constants.GENERATOR_BACKEND_ID, "ollama_chat_default"]:
                api_key_to_use = None  # Ollama doesn't use API keys

            if backend_id in ["gemini_chat_default", "gpt_chat_default"] and not api_key_to_use:
                err_msg = f"{backend_id.split('_')[0].upper()} API Key not found. Set in .env"
                logger.error(err_msg)
                if backend_id == self._active_chat_backend_id:
                    self._is_chat_backend_configured = False
                elif backend_id == self._active_specialized_backend_id:
                    self._is_specialized_backend_configured = False
                self._event_bus.backendConfigurationChanged.emit(backend_id, model_name, False, [])
                self._emit_status_update(err_msg, "#FF6B6B")
                return

            # Configure backend (this will now be non-blocking due to our backend coordinator fixes)
            self._backend_coordinator.configure_backend(backend_id=backend_id, api_key=api_key_to_use,
                                                        model_name=model_name, system_prompt=actual_system_prompt)

        except Exception as e:
            logger.error(f"CM: Error configuring backend '{backend_id}': {e}", exc_info=True)
            self._emit_status_update(f"Failed to configure {backend_id}: {e}", "#FF6B6B")

    def _detect_file_creation_intent(self, user_text: str) -> Optional[str]:
        """Detect if user wants to create a specific file and return the filename"""
        file_creation_patterns = [
            r"create (?:a )?file called ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"create ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"make (?:a )?file ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"write (?:a )?file called ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"generate ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"save (?:this )?as ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
        ]

        for pattern in file_creation_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                filename = match.group(1)
                logger.info(f"Detected file creation intent for: {filename}")
                return filename

        return None

    def _detect_task_type(self, user_text: str, filename: str) -> str:
        """Detect what type of coding task this is based on user text and filename"""
        user_lower = user_text.lower()
        filename_lower = filename.lower()

        # API-related patterns
        api_patterns = [
            r'\b(api|endpoint|route|fastapi|flask|django|rest|http|get|post|put|delete)\b',
            r'\b(server|web service|microservice|backend)\b'
        ]

        # Data processing patterns
        data_patterns = [
            r'\b(data|csv|json|xml|parse|process|analyze|transform|clean)\b',
            r'\b(pandas|numpy|database|sql|query|etl)\b'
        ]

        # UI patterns
        ui_patterns = [
            r'\b(ui|interface|window|dialog|widget|button|form|gui)\b',
            r'\b(pyside|pyqt|tkinter|qt|frontend)\b'
        ]

        # Check filename hints
        if any(term in filename_lower for term in ['api', 'server', 'endpoint', 'route']):
            return 'api'
        elif any(term in filename_lower for term in ['data', 'process', 'parse', 'etl']):
            return 'data_processing'
        elif any(term in filename_lower for term in ['ui', 'window', 'dialog', 'widget', 'gui']):
            return 'ui'
        elif any(term in filename_lower for term in ['util', 'helper', 'tool', 'lib']):
            return 'utility'

        # Check user text patterns
        for pattern in api_patterns:
            if re.search(pattern, user_lower):
                return 'api'

        for pattern in data_patterns:
            if re.search(pattern, user_lower):
                return 'data_processing'

        for pattern in ui_patterns:
            if re.search(pattern, user_lower):
                return 'ui'

        return 'general'  # Default fallback

    def _get_specialized_prompt(self, task_type: str, user_text: str, filename: str) -> str:
        """Get specialized prompt based on detected task type"""

        base_instruction = f"""You are Devstral, an expert coding assistant. The user wants you to create a file called '{filename}' based on their request.

User request: {user_text}

"""

        if task_type == 'api':
            return base_instruction + constants.API_DEVELOPMENT_PROMPT + f"""

Create complete, production-ready code for '{filename}' that implements the requested API functionality."""

        elif task_type == 'data_processing':
            return base_instruction + constants.DATA_PROCESSING_PROMPT + f"""

Create complete, production-ready code for '{filename}' that handles the requested data processing."""

        elif task_type == 'ui':
            return base_instruction + constants.UI_DEVELOPMENT_PROMPT + f"""

Create complete, production-ready code for '{filename}' that implements the requested UI."""

        elif task_type == 'utility':
            return base_instruction + constants.UTILITY_DEVELOPMENT_PROMPT + f"""

Create complete, production-ready code for '{filename}' that provides the requested utility functionality."""

        else:  # general
            return base_instruction + constants.GENERAL_CODING_PROMPT + f"""

Create complete, production-ready code for '{filename}' that fulfills the user's request."""

    def _create_file_creation_prompt(self, user_text: str, filename: str) -> str:
        """Create a specialized prompt for file creation with task-specific optimization"""
        task_type = self._detect_task_type(user_text, filename)
        logger.info(f"CM: Detected task type '{task_type}' for file '{filename}'")
        return self._get_specialized_prompt(task_type, user_text, filename)

    @Slot(str, str, bool, list)
    def _handle_backend_reconfigured_event(self, backend_id: str, model_name: str, is_configured: bool,
                                           available_models: list):
        is_active_chat = (backend_id == self._active_chat_backend_id and model_name == self._active_chat_model_name)
        is_active_spec = (
                backend_id == self._active_specialized_backend_id and model_name == self._active_specialized_model_name)
        if backend_id == self._active_chat_backend_id:
            self._is_chat_backend_configured = is_configured
        elif backend_id == self._active_specialized_backend_id:
            self._is_specialized_backend_configured = is_configured

        # Status updates are emitted by this event, no need to hide loader here as _configure_backend handles it.
        if is_active_chat:
            if is_configured:
                self._emit_status_update(f"Ready. Using {self._active_chat_model_name}", "#98c379", False)
            else:
                self._emit_status_update(
                    f"Failed to config Chat LLM: {self._backend_coordinator.get_last_error_for_backend(backend_id) or 'Unknown'}",
                    "#FF6B6B")
        elif is_active_spec:
            if is_configured:
                self._emit_status_update(f"Specialized LLM {model_name} ready.", "#98c379", True, 3000)
            else:
                self._emit_status_update(
                    f"Specialized LLM Error: {self._backend_coordinator.get_last_error_for_backend(backend_id) or 'Unknown'}",
                    "#FF6B6B", True, 5000)
        self._check_rag_readiness_and_emit_status()

    @Slot(str, list)
    def handle_user_message(self, text: str, image_data: List[Dict[str, Any]]):
        if not self._current_project_id or not self._current_session_id:
            self._emit_status_update("Error: No active project or session.", "#FF6B6B", True, 3000)
            return
        logger.info(
            f"CM: User message for P:{self._current_project_id}/S:{self._current_session_id} - '{text[:50]}...'")

        processed_input = self._user_input_handler.process_input(text, image_data)
        user_msg_txt = processed_input.original_query.strip()
        if not user_msg_txt and not (image_data and processed_input.intent == UserInputIntent.NORMAL_CHAT): return

        if self._current_llm_request_id and processed_input.intent in [UserInputIntent.NORMAL_CHAT,
                                                                       UserInputIntent.FILE_CREATION_REQUEST]:
            self._emit_status_update("Please wait for the current AI response.", "#e5c07b", True, 2500)
            return

        user_msg_parts = [user_msg_txt] if user_msg_txt else []
        if image_data: user_msg_parts.extend(image_data)
        user_message = ChatMessage(role=USER_ROLE, parts=user_msg_parts)
        self._current_chat_history.append(user_message)
        self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id, user_message)
        self._log_llm_comm("USER", user_msg_txt)

        if processed_input.intent == UserInputIntent.NORMAL_CHAT:
            asyncio.create_task(self._handle_normal_chat_async(user_msg_txt, image_data))
        elif processed_input.intent == UserInputIntent.FILE_CREATION_REQUEST:
            self._handle_file_creation_request(user_msg_txt, processed_input.data.get('filename'))
        elif processed_input.intent == UserInputIntent.PLAN_THEN_CODE_REQUEST:
            self._handle_plan_then_code_request(user_msg_txt)  # Loader handled by PlanAndCodeCoordinator
        else:
            unknown_intent_msg = ChatMessage(role=ERROR_ROLE,
                                             parts=[f"[System: Unknown request type: {user_msg_txt[:50]}...]"])
            self._current_chat_history.append(unknown_intent_msg)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              unknown_intent_msg)

    async def _handle_normal_chat_async(self, user_msg_txt: str, image_data: List[Dict[str, Any]]):
        """Handle normal chat interaction with async RAG support"""
        self._event_bus.showLoader.emit("Thinking...")
        if not self._is_chat_backend_configured:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,
                                      parts=["[Error: Chat Backend not ready.]"])
            self._current_chat_history.append(err_msg_obj)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update("Cannot send: Chat AI backend not configured.", "#FF6B6B")
            self._event_bus.hideLoader.emit()
            return

        history_for_llm = self._current_chat_history[:]
        rag_context_str = ""

        should_use_rag = self._rag_handler.should_perform_rag(user_msg_txt, self._is_rag_ready,
                                                              self._is_rag_ready)

        if should_use_rag and self._upload_service:
            self._event_bus.updateLoaderMessage.emit("Searching context...")
            if not self._upload_service._embedder_ready:
                self._emit_status_update("Waiting for RAG system to initialize...", "#e5c07b", False)
                embedder_ready = await self._upload_service.wait_for_embedder_ready(timeout_seconds=10.0)
                if not embedder_ready:
                    self._emit_status_update("RAG system not ready, continuing without context...", "#e5c07b", True,
                                             3000)
                    should_use_rag = False

        if should_use_rag and self._rag_handler:
            logger.info(f"CM: Performing RAG for query: '{user_msg_txt[:50]}'")
            query_entities = self._rag_handler.extract_code_entities(user_msg_txt)
            rag_context_str, queried_collections = self._rag_handler.get_formatted_context(
                query=user_msg_txt,
                query_entities=query_entities,
                project_id=self._current_project_id,
                explicit_focus_paths=[],
                implicit_focus_paths=[],
                is_modification_request=False
            )
            if rag_context_str:
                logger.info(f"CM: RAG context found from collections: {queried_collections}. Prepending to prompt.")
                rag_system_message = ChatMessage(role=SYSTEM_ROLE, parts=[rag_context_str],
                                                 metadata={"is_rag_context": True,
                                                           "queried_collections": queried_collections})
                history_for_llm.insert(-1, rag_system_message)
                rag_notification_msg = ChatMessage(id=uuid.uuid4().hex, role=SYSTEM_ROLE,
                                                   parts=[
                                                       f"[System: RAG used. Context from: {', '.join(queried_collections)}.]"],
                                                   metadata={"is_internal": True})
                self._current_chat_history.append(rag_notification_msg)
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              rag_notification_msg)
                self._log_llm_comm("RAG Context", rag_context_str)
            else:
                logger.info("CM: No relevant RAG context found for query.")

        self._event_bus.updateLoaderMessage.emit(f"Sending to {self._active_chat_model_name}...")
        success, err, req_id = self._backend_coordinator.initiate_llm_chat_request(self._active_chat_backend_id,
                                                                                   history_for_llm, {
                                                                                       "temperature": self._active_temperature})
        if not success or not req_id:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,
                                      parts=[f"[Error sending: {err}]"])
            self._current_chat_history.append(err_msg_obj)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update(f"Failed to start chat: {err or 'Unknown'}", "#FF6B6B")
            self._event_bus.hideLoader.emit()
            return
        self._current_llm_request_id = req_id
        placeholder = ChatMessage(id=req_id, role=MODEL_ROLE, parts=[""],
                                  loading_state=MessageLoadingState.LOADING)
        self._current_chat_history.append(placeholder)
        self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                      placeholder)
        # self._emit_status_update(f"Sending to {self._active_chat_model_name}...", "#61afef") # Now part of loader
        self._backend_coordinator.start_llm_streaming_task(req_id, self._active_chat_backend_id, history_for_llm,
                                                           False, {"temperature": self._active_temperature},
                                                           {"purpose": "normal_chat",
                                                            "project_id": self._current_project_id,
                                                            "session_id": self._current_session_id})

    def _handle_file_creation_request(self, user_msg_txt: str, filename: Optional[str]):
        """Handle direct file creation request from chat"""
        if not self._is_chat_backend_configured:
            self._emit_status_update("Chat backend not configured for file creation.", "#e06c75", True, 3000)
            return

        if not filename:
            filename = self._detect_file_creation_intent(user_msg_txt)
            if not filename:
                asyncio.create_task(self._handle_normal_chat_async(user_msg_txt, []))
                return

        self._event_bus.showLoader.emit(f"Crafting {filename}...")
        logger.info(f"CM: Handling file creation request for '{filename}'")

        file_creation_prompt = self._create_file_creation_prompt(user_msg_txt, filename)
        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[file_creation_prompt])]

        success, err, req_id = self._backend_coordinator.initiate_llm_chat_request(
            self._active_chat_backend_id,
            history_for_llm,
            {"temperature": 0.2}
        )

        if not success or not req_id:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,
                                      parts=[f"[Error creating file: {err}]"])
            self._current_chat_history.append(err_msg_obj)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update(f"Failed to start file creation: {err or 'Unknown'}", "#FF6B6B")
            self._event_bus.hideLoader.emit()
            return

        self._file_creation_request_ids[req_id] = filename
        self._current_llm_request_id = req_id

        placeholder = ChatMessage(id=req_id, role=MODEL_ROLE, parts=[""],
                                  loading_state=MessageLoadingState.LOADING)
        self._current_chat_history.append(placeholder)
        self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                      placeholder)
        # self._emit_status_update(f"Creating {filename}...", "#61afef") # Now part of loader

        self._backend_coordinator.start_llm_streaming_task(
            req_id,
            self._active_chat_backend_id,
            history_for_llm,
            False,
            {"temperature": 0.2},
            {
                "purpose": "file_creation",
                "project_id": self._current_project_id,
                "session_id": self._current_session_id,
                "filename": filename
            }
        )

    def _handle_plan_then_code_request(self, user_msg_txt: str):
        """Handle plan-then-code workflow"""
        if not self._is_chat_backend_configured or not self._is_specialized_backend_configured:
            self._emit_status_update("Planner or Code LLM not configured.", "#e06c75", True, 5000)
            return

        current_project_dir = self._get_current_project_directory()

        # Detect the overall task type for the project
        task_type = self._detect_task_type(user_msg_txt, "multi_file_project")

        # Loader for plan-then-code is handled within PlanAndCodeCoordinator
        self._plan_and_code_coordinator.start_planning_sequence(
            user_query=user_msg_txt,
            planner_llm_backend_id=self._active_chat_backend_id,
            planner_llm_model_name=self._active_chat_model_name,
            planner_llm_temperature=self._active_temperature,
            specialized_llm_backend_id=self._active_specialized_backend_id,
            specialized_llm_model_name=self._active_specialized_model_name,
            project_files_dir=current_project_dir,
            project_id=self._current_project_id,
            session_id=self._current_session_id,
            user_task_type=task_type
        )

    def _get_current_project_directory(self) -> str:
        """Get the current project's working directory with smart fallbacks"""
        import os
        from datetime import datetime

        user_projects_dir = getattr(self, '_user_projects_directory', None)
        if user_projects_dir and os.path.exists(user_projects_dir):
            return user_projects_dir

        base_output_dir = os.path.join(os.getcwd(), "ava_generated_projects")

        if self._current_project_id and self._project_manager:
            project_obj = self._project_manager.get_project_by_id(self._current_project_id)
            if project_obj:
                safe_project_name = "".join(c for c in project_obj.name if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_project_name = safe_project_name.replace(' ', '_')
                project_output_dir = os.path.join(base_output_dir, safe_project_name)
            else:
                project_output_dir = os.path.join(base_output_dir, f"project_{self._current_project_id[:8]}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_output_dir = os.path.join(base_output_dir, f"untitled_project_{timestamp}")

        os.makedirs(project_output_dir, exist_ok=True)
        logger.info(f"CM: Using project directory: {project_output_dir}")

        return project_output_dir

    def _extract_code_from_response(self, response_text: str) -> Optional[str]:
        """Extract code from LLM response, handling various code block formats"""
        python_pattern = r"```python\s*\n(.*?)```"
        match = re.search(python_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        generic_pattern = r"```.*?\n(.*?)```"
        match = re.search(generic_pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    @Slot()
    def request_new_chat_session(self):
        logger.info("CM: New chat session requested by user/UI.")
        if not self._current_project_id:
            self._emit_status_update("Cannot start new chat: No active project.", "#e06c75", True, 3000)
            return
        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(request_id=self._current_llm_request_id)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._event_bus.hideLoader.emit()  # Ensure loader is hidden
        self._event_bus.createNewSessionForProjectRequested.emit(self._current_project_id)

    @Slot(str)
    def request_global_rag_scan_directory(self, dir_path: str):
        if not self._upload_service or not self._upload_service.is_vector_db_ready(
                constants.GLOBAL_COLLECTION_ID):
            self._emit_status_update(f"RAG system not ready for Global Knowledge. Cannot scan.", "#FF6B6B", True, 4000)
            return

        logger.info(f"CM: Requesting GLOBAL RAG scan for directory: {dir_path}")
        self._event_bus.showLoader.emit(f"Scanning '{os.path.basename(dir_path)}' for Global RAG...")
        rag_message: Optional[ChatMessage] = None
        try:
            rag_message = self._upload_service.process_directory_for_context(dir_path,
                                                                             collection_id=constants.GLOBAL_COLLECTION_ID)
        finally:
            self._event_bus.hideLoader.emit()

        if rag_message and self._current_project_id and self._current_session_id:
            self._current_chat_history.append(rag_message)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          rag_message)
            self._log_llm_comm(f"RAG_SCAN_GLOBAL", rag_message.text)
            if rag_message.role == ERROR_ROLE:
                self._emit_status_update(f"Global RAG Scan completed with errors.", "#FF6B6B", True, 5000)
            else:
                self._emit_status_update(f"Global RAG Scan complete!", "#98c379", True, 3000)
        elif rag_message:
            logger.info(
                f"Global RAG Scan completed. Message (not added to active chat): {rag_message.text}")
        else:
            self._emit_status_update(f"Global RAG Scan failed or returned no message.", "#FF6B6B", True, 3000)

        self._check_rag_readiness_and_emit_status()

    @Slot(list, str)
    def handle_project_files_upload_request(self, file_paths: List[str], project_id: str):
        if not project_id:
            logger.error("CM: Project RAG file upload requested without a project_id.")
            self._emit_status_update("Cannot add files to project RAG: Project ID missing.", "#e06c75", True, 3000)
            return

        if not self._upload_service or not self._upload_service.is_vector_db_ready(project_id):
            self._emit_status_update(f"RAG system not ready for project '{project_id[:8]}...'. Cannot add files.",
                                     "#FF6B6B", True, 4000)
            return

        logger.info(f"CM: Requesting Project RAG file upload for {len(file_paths)} files, project: {project_id}")
        self._event_bus.showLoader.emit(f"Adding {len(file_paths)} files to RAG for '{project_id[:8]}...'")
        rag_message: Optional[ChatMessage] = None
        try:
            rag_message = self._upload_service.process_files_for_context(file_paths,
                                                                         collection_id=project_id)
        finally:
            self._event_bus.hideLoader.emit()

        if rag_message and self._current_project_id == project_id and self._current_session_id:
            self._current_chat_history.append(rag_message)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          rag_message)
            self._log_llm_comm(f"RAG_UPLOAD_P:{project_id[:6]}", rag_message.text)
            if rag_message.role == ERROR_ROLE:
                self._emit_status_update(f"Project RAG file add for '{project_id[:8]}' completed with errors.",
                                         "#FF6B6B", True, 5000)
            else:
                self._emit_status_update(f"Project RAG file add for '{project_id[:8]}' complete!", "#98c379", True,
                                         3000)
        elif rag_message:
            logger.info(
                f"Project RAG file add for project '{project_id}' completed. Message (not added to active chat): {rag_message.text}")
            self._emit_status_update(f"Files added to RAG for (non-active) project '{project_id[:8]}'.", "#56b6c2",
                                     True, 4000)
        else:
            self._emit_status_update(f"Project RAG file add for '{project_id[:8]}' failed or returned no message.",
                                     "#FF6B6B", True, 3000)

        self._check_rag_readiness_and_emit_status()

    @Slot(str, str)
    def _handle_llm_request_sent(self, backend_id: str, request_id: str):
        if request_id == self._current_llm_request_id and backend_id == self._active_chat_backend_id:
            self._event_bus.uiInputBarBusyStateChanged.emit(True)
            # Loader is already shown by the methods initiating the request

    @Slot(str, object, dict)
    def _handle_llm_response_completed(self, request_id: str, completed_message_obj: object, usage_stats_dict: dict):
        meta_pid = usage_stats_dict.get("project_id")
        meta_sid = usage_stats_dict.get("session_id")
        purpose = usage_stats_dict.get("purpose")

        if request_id == self._current_llm_request_id and meta_pid == self._current_project_id and meta_sid == self._current_session_id:
            self._event_bus.hideLoader.emit()  # Hide loader for this request
            if isinstance(completed_message_obj, ChatMessage):
                self._log_llm_comm(self._active_chat_backend_id.upper() + " RESPONSE",
                                   completed_message_obj.text)

                if purpose == "file_creation" and request_id in self._file_creation_request_ids:
                    filename = self._file_creation_request_ids.pop(request_id)
                    extracted_code = self._extract_code_from_response(completed_message_obj.text)

                    if extracted_code:
                        logger.info(f"CM: Extracted code for file '{filename}', sending to code viewer")
                        self._event_bus.modificationFileReadyForDisplay.emit(filename, extracted_code)
                        file_creation_msg = ChatMessage(
                            id=uuid.uuid4().hex,
                            role=SYSTEM_ROLE,
                            parts=[f"[File created: {filename} - View in Code Viewer]"],
                            metadata={"is_file_creation": True, "filename": filename}
                        )
                        self._current_chat_history.append(file_creation_msg)
                        self._event_bus.newMessageAddedToHistory.emit(
                            self._current_project_id,
                            self._current_session_id,
                            file_creation_msg
                        )
                    else:
                        logger.warning(f"CM: Could not extract code from response for file '{filename}'")

                updated = False
                for i, msg in enumerate(self._current_chat_history):
                    if msg.id == request_id:
                        self._current_chat_history[i] = completed_message_obj
                        self._current_chat_history[i].loading_state = MessageLoadingState.COMPLETED
                        updated = True;
                        break
                if not updated: self._current_chat_history.append(completed_message_obj)
                if self._current_project_id and self._current_session_id:
                    self._event_bus.messageFinalizedForSession.emit(self._current_project_id, self._current_session_id,
                                                                    request_id, completed_message_obj, usage_stats_dict,
                                                                    False)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._emit_status_update(f"Ready. Last: {self._active_chat_model_name}", "#98c379")
            self._check_rag_readiness_and_emit_status()

    @Slot(str, str)
    def _handle_llm_response_error(self, request_id: str, error_message_str: str):
        if request_id == self._current_llm_request_id:
            self._event_bus.hideLoader.emit()  # Hide loader for this request
            self._log_llm_comm(f"{self._active_chat_backend_id.upper()} ERROR", error_message_str)

            if request_id in self._file_creation_request_ids:
                filename = self._file_creation_request_ids.pop(request_id)
                logger.warning(f"CM: File creation failed for '{filename}': {error_message_str}")

            err_chat_msg = ChatMessage(id=request_id, role=ERROR_ROLE, parts=[f"[AI Error: {error_message_str}]"],
                                       loading_state=MessageLoadingState.ERROR)
            updated = False
            for i, msg in enumerate(self._current_chat_history):
                if msg.id == request_id: self._current_chat_history[
                    i] = err_chat_msg; updated = True; break
            if not updated: self._current_chat_history.append(err_chat_msg)
            if self._current_project_id and self._current_session_id:
                self._event_bus.messageFinalizedForSession.emit(self._current_project_id, self._current_session_id,
                                                                request_id, err_chat_msg, {}, True)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._emit_status_update(f"Error: {error_message_str[:60]}...", "#FF6B6B")
            self._check_rag_readiness_and_emit_status()

    @Slot(str, str)
    def _handle_chat_llm_selection_event(self, backend_id: str, model_name: str):
        if self._active_chat_backend_id != backend_id or self._active_chat_model_name != model_name:
            self._active_chat_backend_id, self._active_chat_model_name = backend_id, model_name
            self._configure_backend(backend_id, model_name, self._active_chat_personality_prompt)

    @Slot(str, str)
    def _handle_specialized_llm_selection_event(self, backend_id: str, model_name: str):
        if self._active_specialized_backend_id != backend_id or self._active_specialized_model_name != model_name:
            self._active_specialized_backend_id, self._active_specialized_model_name = backend_id, model_name
            self._configure_backend(backend_id, model_name, constants.CODER_AI_SYSTEM_PROMPT)

    @Slot(str, str)
    def _handle_personality_change_event(self, new_prompt: str, backend_id_for_persona: str):
        if backend_id_for_persona == self._active_chat_backend_id:
            self._active_chat_personality_prompt = new_prompt.strip() if new_prompt and new_prompt.strip() else None
            self._configure_backend(self._active_chat_backend_id, self._active_chat_model_name,
                                    self._active_chat_personality_prompt)

    def _check_rag_readiness_and_emit_status(self):
        if not self._upload_service or not hasattr(self._upload_service,
                                                   '_vector_db_service') or not self._upload_service._vector_db_service:
            self._is_rag_ready = False
            self._event_bus.ragStatusChanged.emit(False, "RAG Not Ready (Service Error)", "#e06c75")
            return

        rag_status_message = "RAG Status: Initializing..."
        rag_status_color = constants.TIMESTAMP_COLOR_HEX
        current_context_rag_ready = False

        if not self._upload_service._embedder_ready:
            rag_status_message = "RAG: Initializing embedder..."
            rag_status_color = "#e5c07b"
        elif not self._current_project_id:
            if not self._upload_service._dependencies_ready:
                rag_status_message = "Global RAG: Dependencies Missing"
                rag_status_color = "#e06c75"
            else:
                global_size = self._upload_service._vector_db_service.get_collection_size(
                    constants.GLOBAL_COLLECTION_ID)
                if global_size == -1:
                    rag_status_message = "Global RAG: DB Error"
                    rag_status_color = "#e06c75"
                elif global_size == 0:
                    rag_status_message = "Global RAG: Ready (Empty)"
                    rag_status_color = "#e5c07b"
                    current_context_rag_ready = True
                else:
                    rag_status_message = f"Global RAG: Ready ({global_size} chunks)"
                    rag_status_color = "#98c379"
                    current_context_rag_ready = True
        else:
            project_name_display = self._project_manager.get_project_by_id(self._current_project_id)
            project_display_name = project_name_display.name[:15] if project_name_display else self._current_project_id[
                                                                                               :8]

            if not self._upload_service._dependencies_ready:
                rag_status_message = f"RAG for '{project_display_name}...': Dependencies Missing"
                rag_status_color = "#e06c75"
            else:
                project_size = self._upload_service._vector_db_service.get_collection_size(
                    self._current_project_id)
                if project_size == -1:
                    rag_status_message = f"RAG for '{project_display_name}...': DB Error"
                    rag_status_color = "#e06c75"
                elif project_size == 0:
                    rag_status_message = f"RAG for '{project_display_name}...': Ready (Empty)"
                    rag_status_color = "#e5c07b"
                    current_context_rag_ready = True
                else:
                    rag_status_message = f"RAG for '{project_display_name}...': Ready ({project_size} chunks)"
                    rag_status_color = "#98c379"
                    current_context_rag_ready = True

        self._is_rag_ready = current_context_rag_ready
        logger.info(f"CM: Emitting RAG Status: OverallReady={self._is_rag_ready}, Msg='{rag_status_message}'")
        self._event_bus.ragStatusChanged.emit(self._is_rag_ready, rag_status_message, rag_status_color)

    def set_model_for_backend(self, backend_id: str, model_name: str):
        if backend_id == self._active_chat_backend_id:
            if self._active_chat_model_name != model_name: self._handle_chat_llm_selection_event(backend_id, model_name)
        elif backend_id == self._active_specialized_backend_id:
            if self._active_specialized_model_name != model_name: self._handle_specialized_llm_selection_event(
                backend_id, model_name)

    def set_chat_temperature(self, temperature: float):
        if 0.0 <= temperature <= 2.0: self._active_temperature = temperature; self._emit_status_update(
            f"Temp: {temperature:.2f}", "#61afef", True, 1500)

    def get_current_chat_history(self) -> List[ChatMessage]:
        return self._current_chat_history

    def get_current_project_id(self) -> Optional[str]:
        return self._current_project_id

    def get_current_session_id(self) -> Optional[str]:
        return self._current_session_id

    def get_current_active_chat_backend_id(self) -> str:
        return self._active_chat_backend_id

    def get_current_active_chat_model_name(self) -> str:
        return self._active_chat_model_name

    def get_current_active_specialized_backend_id(self) -> str:
        return self._active_specialized_backend_id

    def get_current_active_specialized_model_name(self) -> str:
        return self._active_specialized_model_name

    def get_model_for_backend(self, backend_id: str) -> Optional[str]:
        if backend_id == self._active_chat_backend_id: return self._active_chat_model_name
        if backend_id == self._active_specialized_backend_id: return self._active_specialized_model_name
        return self._backend_coordinator.get_current_configured_model(backend_id)

    def get_all_available_backend_ids(self) -> List[str]:
        return self._backend_coordinator.get_all_backend_ids()

    def get_available_models_for_backend(self, backend_id: str) -> List[str]:
        return self._backend_coordinator.get_available_models_for_backend(backend_id)

    def get_current_chat_personality(self) -> Optional[str]:
        return self._active_chat_personality_prompt

    def get_current_specialized_personality(self) -> Optional[str]:
        return constants.CODER_AI_SYSTEM_PROMPT

    def get_current_chat_temperature(self) -> float:
        return self._active_temperature

    def is_api_ready(self) -> bool:
        return self._is_chat_backend_configured

    def is_specialized_api_ready(self) -> bool:
        return self._is_specialized_backend_configured

    def is_rag_ready(self) -> bool:
        return self._is_rag_ready

    def is_overall_busy(self) -> bool:
        return bool(self._current_llm_request_id) or self._backend_coordinator.is_any_backend_busy()

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:
        return self._llm_comm_logger

    def get_backend_coordinator(self) -> BackendCoordinator:
        return self._backend_coordinator

    def get_project_manager(self) -> ProjectManager:
        return self._project_manager

    def get_upload_service(self) -> UploadService:
        return self._upload_service

    def get_rag_handler(self) -> RagHandler:
        return self._rag_handler

    def cleanup_phase1(self):
        logger.info("ChatManager cleanup...")
        if self._current_llm_request_id: self._backend_coordinator.cancel_current_task(
            self._current_llm_request_id); self._current_llm_request_id = None
        self._event_bus.hideLoader.emit()  # Ensure loader is hidden on cleanup