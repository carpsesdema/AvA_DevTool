# core/chat_manager.py
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
    ProjectManager = type("ProjectManager", (object,), {})  # type: ignore
    UploadService = type("UploadService", (object,), {})  # type: ignore
    RagHandler = type("RagHandler", (object,), {})  # type: ignore
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

        if not isinstance(self._project_manager, ProjectManager):  # type: ignore
            logger.critical("ChatManager received an invalid ProjectManager instance.")
            raise TypeError("ChatManager requires a valid ProjectManager instance from Orchestrator.")

        self._user_input_handler = UserInputHandler()
        self._plan_and_code_coordinator = PlanAndCodeCoordinator(
            backend_coordinator=self._backend_coordinator,
            event_bus=self._event_bus,
            llm_comm_logger=self._llm_comm_logger,
            parent=self
        )

        self._current_chat_history: List[ChatMessage] = []  # type: ignore
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

        self._connect_event_bus_subscriptions()
        logger.info("ChatManager initialized and subscriptions connected.")

    def initialize(self):
        logger.info("ChatManager late initialization...")
        self._configure_backend(self._active_chat_backend_id, self._active_chat_model_name,
                                self._active_chat_personality_prompt)
        self._configure_backend(self._active_specialized_backend_id, self._active_specialized_model_name,
                                constants.CODER_AI_SYSTEM_PROMPT)
        self._check_rag_readiness_and_emit_status()

    def set_active_session(self, project_id: str, session_id: str, history: List[ChatMessage]):  # type: ignore
        logger.info(f"CM: Setting active session to P:{project_id}/S:{session_id}. History items: {len(history)}")
        old_project_id = self._current_project_id
        self._current_project_id = project_id
        self._current_session_id = session_id
        self._current_chat_history = list(history)

        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(self._current_llm_request_id)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)

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
        else:
            logger.info(f"LLM_COMM_LOG_FALLBACK: [{sender}] {message[:150]}...")

    def _configure_backend(self, backend_id: str, model_name: str, system_prompt: Optional[str]):
        logger.info(f"CM: Configuring backend '{backend_id}' with model '{model_name}'")
        api_key_to_use: Optional[str] = None
        actual_system_prompt = system_prompt
        if backend_id == constants.GENERATOR_BACKEND_ID:  # type: ignore
            actual_system_prompt = constants.CODER_AI_SYSTEM_PROMPT  # type: ignore
            logger.info(f"CM: Using CODER_AI_SYSTEM_PROMPT for {backend_id}")
        if backend_id == "gemini_chat_default":
            api_key_to_use = get_gemini_api_key()
        elif backend_id == "gpt_chat_default":
            api_key_to_use = get_openai_api_key()
        elif backend_id in [constants.GENERATOR_BACKEND_ID, "ollama_chat_default"]:  # type: ignore
            api_key_to_use = None  # Ollama doesn't use API keys in the same way
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
        self._backend_coordinator.configure_backend(backend_id=backend_id, api_key=api_key_to_use,
                                                    model_name=model_name, system_prompt=actual_system_prompt)

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

    def _create_file_creation_prompt(self, user_text: str, filename: str) -> str:
        """Create a specialized prompt for file creation"""
        return f"""You are a helpful coding assistant. The user wants you to create a file called '{filename}' based on their request.

User request: {user_text}

Please generate the complete, functional Python code for '{filename}'. Your response should contain:

1. Complete, working Python code
2. Proper imports
3. Clear function/class definitions with docstrings
4. Type hints where appropriate
5. Error handling where needed
6. Comments explaining complex logic

Format your response as a single Python code block like this:
```python
# {filename}
# Your complete code here
```

Make the code production-ready and well-structured."""

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
        if image_data: user_msg_parts.extend(image_data)  # type: ignore
        user_message = ChatMessage(role=USER_ROLE, parts=user_msg_parts)  # type: ignore
        self._current_chat_history.append(user_message)
        self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id, user_message)
        self._log_llm_comm("USER", user_msg_txt)

        if processed_input.intent == UserInputIntent.NORMAL_CHAT:
            self._handle_normal_chat(user_msg_txt, image_data)
        elif processed_input.intent == UserInputIntent.FILE_CREATION_REQUEST:
            self._handle_file_creation_request(user_msg_txt, processed_input.data.get('filename'))
        elif processed_input.intent == UserInputIntent.PLAN_THEN_CODE_REQUEST:
            self._handle_plan_then_code_request(user_msg_txt)
        else:
            unknown_intent_msg = ChatMessage(role=ERROR_ROLE,  # type: ignore
                                             parts=[f"[System: Unknown request type: {user_msg_txt[:50]}...]"])
            self._current_chat_history.append(unknown_intent_msg)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              unknown_intent_msg)

    def _handle_normal_chat(self, user_msg_txt: str, image_data: List[Dict[str, Any]]):
        """Handle normal chat interaction"""
        if not self._is_chat_backend_configured:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,  # type: ignore
                                      parts=["[Error: Chat Backend not ready.]"])
            self._current_chat_history.append(err_msg_obj)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update("Cannot send: Chat AI backend not configured.", "#FF6B6B")
            return

        history_for_llm = self._current_chat_history[:]
        rag_context_str = ""
        # MODIFICATION: Pass self._is_rag_ready (which reflects current context readiness)
        if self._rag_handler.should_perform_rag(user_msg_txt, self._is_rag_ready,
                                                self._is_rag_ready):  # type: ignore
            logger.info(f"CM: Performing RAG for query: '{user_msg_txt[:50]}'")
            self._emit_status_update("Searching RAG context...", "#61afef", True, 1500)
            query_entities = self._rag_handler.extract_code_entities(user_msg_txt)  # type: ignore
            # MODIFICATION: Pass current project_id for project-specific RAG
            rag_context_str, queried_collections = self._rag_handler.get_formatted_context(  # type: ignore
                query=user_msg_txt,
                query_entities=query_entities,
                project_id=self._current_project_id,  # Explicitly pass current project ID
                explicit_focus_paths=[],
                implicit_focus_paths=[],
                is_modification_request=False
            )
            if rag_context_str:
                logger.info(f"CM: RAG context found from collections: {queried_collections}. Prepending to prompt.")
                rag_system_message = ChatMessage(role=SYSTEM_ROLE, parts=[rag_context_str],  # type: ignore
                                                 metadata={"is_rag_context": True,
                                                           "queried_collections": queried_collections})
                history_for_llm.insert(-1, rag_system_message)  # Insert before the last user message
                rag_notification_msg = ChatMessage(id=uuid.uuid4().hex, role=SYSTEM_ROLE,  # type: ignore
                                                   parts=[
                                                       f"[System: RAG used. Context from: {', '.join(queried_collections)}.]"],
                                                   metadata={"is_internal": True})
                self._current_chat_history.append(rag_notification_msg)
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              rag_notification_msg)
                self._log_llm_comm("RAG Context", rag_context_str)
            else:
                logger.info("CM: No relevant RAG context found for query.")

        success, err, req_id = self._backend_coordinator.initiate_llm_chat_request(self._active_chat_backend_id,
                                                                                   history_for_llm, {
                                                                                       "temperature": self._active_temperature})
        if not success or not req_id:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,
                                      parts=[f"[Error sending: {err}]"])  # type: ignore
            self._current_chat_history.append(err_msg_obj)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update(f"Failed to start chat: {err or 'Unknown'}", "#FF6B6B")
            return
        self._current_llm_request_id = req_id
        placeholder = ChatMessage(id=req_id, role=MODEL_ROLE, parts=[""],
                                  loading_state=MessageLoadingState.LOADING)  # type: ignore
        self._current_chat_history.append(placeholder)
        self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                      placeholder)
        self._emit_status_update(f"Sending to {self._active_chat_model_name}...", "#61afef")
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
            # If no filename detected, try to extract one from the message
            filename = self._detect_file_creation_intent(user_msg_txt)
            if not filename:
                # Fall back to normal chat if we can't determine a filename
                self._handle_normal_chat(user_msg_txt, [])
                return

        logger.info(f"CM: Handling file creation request for '{filename}'")

        # Create specialized prompt for file creation
        file_creation_prompt = self._create_file_creation_prompt(user_msg_txt, filename)
        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[file_creation_prompt])]

        success, err, req_id = self._backend_coordinator.initiate_llm_chat_request(
            self._active_chat_backend_id,
            history_for_llm,
            {"temperature": 0.2}  # Lower temperature for code generation
        )

        if not success or not req_id:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,
                                      parts=[f"[Error creating file: {err}]"])  # type: ignore
            self._current_chat_history.append(err_msg_obj)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update(f"Failed to start file creation: {err or 'Unknown'}", "#FF6B6B")
            return

        # Track this as a file creation request
        self._file_creation_request_ids[req_id] = filename
        self._current_llm_request_id = req_id

        placeholder = ChatMessage(id=req_id, role=MODEL_ROLE, parts=[""],
                                  loading_state=MessageLoadingState.LOADING)  # type: ignore
        self._current_chat_history.append(placeholder)
        self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                      placeholder)
        self._emit_status_update(f"Creating {filename}...", "#61afef")

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

        # Get current project directory for file operations
        current_project_dir = self._get_current_project_directory()

        self._plan_and_code_coordinator.start_planning_sequence(  # type: ignore
            user_query=user_msg_txt,
            planner_llm_backend_id=self._active_chat_backend_id,
            planner_llm_model_name=self._active_chat_model_name,
            planner_llm_temperature=self._active_temperature,
            specialized_llm_backend_id=self._active_specialized_backend_id,
            specialized_llm_model_name=self._active_specialized_model_name,
            project_files_dir=current_project_dir,
            project_id=self._current_project_id,  # NEW: Pass project context
            session_id=self._current_session_id  # NEW: Pass session context
        )

    def _get_current_project_directory(self) -> str:
        """Get the current project's working directory"""
        # For now, return current working directory
        # This could be enhanced to use project-specific directories
        return os.getcwd()

    def _extract_code_from_response(self, response_text: str) -> Optional[str]:
        """Extract code from LLM response, handling various code block formats"""
        # Try to find Python code blocks first
        python_pattern = r"```python\s*\n(.*?)```"
        match = re.search(python_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try generic code blocks
        generic_pattern = r"```.*?\n(.*?)```"
        match = re.search(generic_pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If no code blocks found, return None
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
        self._event_bus.createNewSessionForProjectRequested.emit(self._current_project_id)

    @Slot(str)  # MODIFICATION: Slot for global RAG scan, only dir_path
    def request_global_rag_scan_directory(self, dir_path: str):
        if not self._upload_service or not self._upload_service.is_vector_db_ready(
                constants.GLOBAL_COLLECTION_ID):  # type: ignore
            self._emit_status_update(f"RAG system not ready for Global Knowledge. Cannot scan.", "#FF6B6B", True, 4000)
            return

        logger.info(f"CM: Requesting GLOBAL RAG scan for directory: {dir_path}")
        self._emit_status_update(f"Scanning '{os.path.basename(dir_path)}' for Global RAG...", "#61afef", False)

        # Call UploadService with GLOBAL_COLLECTION_ID
        rag_message = self._upload_service.process_directory_for_context(dir_path,
                                                                         collection_id=constants.GLOBAL_COLLECTION_ID)  # type: ignore

        if rag_message and self._current_project_id and self._current_session_id:  # Add to current active chat
            self._current_chat_history.append(rag_message)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          rag_message)
            self._log_llm_comm(f"RAG_SCAN_GLOBAL", rag_message.text)  # type: ignore
            if rag_message.role == ERROR_ROLE:  # type: ignore
                self._emit_status_update(f"Global RAG Scan completed with errors.", "#FF6B6B", True, 5000)
            else:
                self._emit_status_update(f"Global RAG Scan complete!", "#98c379", True, 3000)
        elif rag_message:
            logger.info(
                f"Global RAG Scan completed. Message (not added to active chat): {rag_message.text}")  # type: ignore
            # Status update for non-active chat context if needed
        else:
            self._emit_status_update(f"Global RAG Scan failed or returned no message.", "#FF6B6B", True, 3000)

        self._check_rag_readiness_and_emit_status()  # Re-check RAG status

    @Slot(list, str)  # NEW SLOT: For project-specific file uploads
    def handle_project_files_upload_request(self, file_paths: List[str], project_id: str):
        if not project_id:
            logger.error("CM: Project RAG file upload requested without a project_id.")
            self._emit_status_update("Cannot add files to project RAG: Project ID missing.", "#e06c75", True, 3000)
            return

        if not self._upload_service or not self._upload_service.is_vector_db_ready(project_id):  # type: ignore
            self._emit_status_update(f"RAG system not ready for project '{project_id[:8]}...'. Cannot add files.",
                                     "#FF6B6B", True, 4000)
            return

        logger.info(f"CM: Requesting Project RAG file upload for {len(file_paths)} files, project: {project_id}")
        self._emit_status_update(f"Adding {len(file_paths)} files to RAG for project '{project_id[:8]}...'...",
                                 "#61afef", False)

        # Call UploadService with the specific project_id as collection_id
        rag_message = self._upload_service.process_files_for_context(file_paths,
                                                                     collection_id=project_id)  # type: ignore

        # Add confirmation/error message to the chat of the *active* project if it matches the target project
        if rag_message and self._current_project_id == project_id and self._current_session_id:
            self._current_chat_history.append(rag_message)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          rag_message)
            self._log_llm_comm(f"RAG_UPLOAD_P:{project_id[:6]}", rag_message.text)  # type: ignore
            if rag_message.role == ERROR_ROLE:  # type: ignore
                self._emit_status_update(f"Project RAG file add for '{project_id[:8]}' completed with errors.",
                                         "#FF6B6B", True, 5000)
            else:
                self._emit_status_update(f"Project RAG file add for '{project_id[:8]}' complete!", "#98c379", True,
                                         3000)
        elif rag_message:  # Files were added to a non-active project's RAG
            logger.info(
                f"Project RAG file add for project '{project_id}' completed. Message (not added to active chat): {rag_message.text}")  # type: ignore
            self._emit_status_update(f"Files added to RAG for (non-active) project '{project_id[:8]}'.", "#56b6c2",
                                     True, 4000)
        else:
            self._emit_status_update(f"Project RAG file add for '{project_id[:8]}' failed or returned no message.",
                                     "#FF6B6B", True, 3000)

        self._check_rag_readiness_and_emit_status()  # Re-check RAG status

    @Slot(str, str)
    def _handle_llm_request_sent(self, backend_id: str, request_id: str):
        if request_id == self._current_llm_request_id and backend_id == self._active_chat_backend_id:
            self._event_bus.uiInputBarBusyStateChanged.emit(True)

    @Slot(str, object, dict)
    def _handle_llm_response_completed(self, request_id: str, completed_message_obj: object, usage_stats_dict: dict):
        meta_pid = usage_stats_dict.get("project_id")
        meta_sid = usage_stats_dict.get("session_id")
        purpose = usage_stats_dict.get("purpose")

        if request_id == self._current_llm_request_id and meta_pid == self._current_project_id and meta_sid == self._current_session_id:
            if isinstance(completed_message_obj, ChatMessage):  # type: ignore
                self._log_llm_comm(self._active_chat_backend_id.upper() + " RESPONSE",
                                   completed_message_obj.text)  # type: ignore

                # Handle file creation requests
                if purpose == "file_creation" and request_id in self._file_creation_request_ids:
                    filename = self._file_creation_request_ids.pop(request_id)
                    extracted_code = self._extract_code_from_response(completed_message_obj.text)  # type: ignore

                    if extracted_code:
                        logger.info(f"CM: Extracted code for file '{filename}', sending to code viewer")
                        # Send the generated file to the code viewer
                        self._event_bus.modificationFileReadyForDisplay.emit(filename, extracted_code)

                        # Add a system message to chat about the file creation
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
                    if msg.id == request_id:  # type: ignore
                        self._current_chat_history[i] = completed_message_obj  # type: ignore
                        self._current_chat_history[i].loading_state = MessageLoadingState.COMPLETED  # type: ignore
                        updated = True;
                        break
                if not updated: self._current_chat_history.append(completed_message_obj)  # type: ignore
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
            self._log_llm_comm(f"{self._active_chat_backend_id.upper()} ERROR", error_message_str)

            # Clean up file creation tracking if this was a file creation request
            if request_id in self._file_creation_request_ids:
                filename = self._file_creation_request_ids.pop(request_id)
                logger.warning(f"CM: File creation failed for '{filename}': {error_message_str}")

            err_chat_msg = ChatMessage(id=request_id, role=ERROR_ROLE, parts=[f"[AI Error: {error_message_str}]"],
                                       # type: ignore
                                       loading_state=MessageLoadingState.ERROR)
            updated = False
            for i, msg in enumerate(self._current_chat_history):
                if msg.id == request_id: self._current_chat_history[
                    i] = err_chat_msg; updated = True; break  # type: ignore
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
            self._configure_backend(backend_id, model_name, constants.CODER_AI_SYSTEM_PROMPT)  # type: ignore

    @Slot(str, str)
    def _handle_personality_change_event(self, new_prompt: str, backend_id_for_persona: str):
        if backend_id_for_persona == self._active_chat_backend_id:
            self._active_chat_personality_prompt = new_prompt.strip() if new_prompt and new_prompt.strip() else None
            self._configure_backend(self._active_chat_backend_id, self._active_chat_model_name,
                                    self._active_chat_personality_prompt)

    def _check_rag_readiness_and_emit_status(self):
        # MODIFICATION: Fixed RAG readiness check to not treat non-existent collections as errors
        if not self._upload_service or not hasattr(self._upload_service,
                                                   '_vector_db_service') or not self._upload_service._vector_db_service:  # type: ignore
            self._is_rag_ready = False
            self._event_bus.ragStatusChanged.emit(False, "RAG Not Ready (Service Error)", "#e06c75")
            return

        rag_status_message = "RAG Status: Initializing..."
        rag_status_color = constants.TIMESTAMP_COLOR_HEX  # type: ignore
        current_context_rag_ready = False

        if not self._current_project_id:  # No active project, check GLOBAL RAG
            if not self._upload_service._dependencies_ready:  # type: ignore
                rag_status_message = "Global RAG: Dependencies Missing"
                rag_status_color = "#e06c75"
            else:
                # Try to get collection size, which will create the collection if it doesn't exist
                global_size = self._upload_service._vector_db_service.get_collection_size(
                    constants.GLOBAL_COLLECTION_ID)  # type: ignore
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
        else:  # Active project, check project-specific RAG
            project_name_display = self._project_manager.get_project_by_id(self._current_project_id)  # type: ignore
            project_display_name = project_name_display.name[:15] if project_name_display else self._current_project_id[
                                                                                               :8]

            if not self._upload_service._dependencies_ready:  # type: ignore
                rag_status_message = f"RAG for '{project_display_name}...': Dependencies Missing"
                rag_status_color = "#e06c75"
            else:
                # Try to get collection size, which will create the collection if it doesn't exist
                project_size = self._upload_service._vector_db_service.get_collection_size(
                    self._current_project_id)  # type: ignore
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

        self._is_rag_ready = current_context_rag_ready  # Update the general flag based on current context
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

    def get_current_chat_history(self) -> List[ChatMessage]:  # type: ignore
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
        return constants.CODER_AI_SYSTEM_PROMPT  # type: ignore

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

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:  # type: ignore
        return self._llm_comm_logger

    def get_backend_coordinator(self) -> BackendCoordinator:  # type: ignore
        return self._backend_coordinator

    def get_project_manager(self) -> ProjectManager:  # type: ignore
        return self._project_manager

    def get_upload_service(self) -> UploadService:  # type: ignore
        return self._upload_service  # type: ignore

    def get_rag_handler(self) -> RagHandler:  # type: ignore
        return self._rag_handler  # type: ignore

    def cleanup_phase1(self):
        logger.info("ChatManager cleanup...")
        if self._current_llm_request_id: self._backend_coordinator.cancel_current_task(
            self._current_llm_request_id); self._current_llm_request_id = None