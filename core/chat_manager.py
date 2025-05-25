# core/chat_manager.py
import logging
import os
import uuid
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from PySide6.QtCore import QObject, Slot

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
    def __init__(self, orchestrator: 'ApplicationOrchestrator', parent: Optional[QObject] = None):
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
        self._is_rag_ready: bool = False
        self._current_llm_request_id: Optional[str] = None

        self._connect_event_bus_subscriptions()
        logger.info("ChatManager initialized and subscriptions connected.")

    def initialize(self):
        logger.info("ChatManager late initialization...")
        self._configure_backend(self._active_chat_backend_id, self._active_chat_model_name,
                                self._active_chat_personality_prompt)
        self._configure_backend(self._active_specialized_backend_id, self._active_specialized_model_name,
                                constants.CODER_AI_SYSTEM_PROMPT)
        self._check_rag_readiness_and_emit_status()

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

        self._event_bus.activeSessionHistoryCleared.emit(project_id, session_id)
        self._event_bus.activeSessionHistoryLoaded.emit(project_id, session_id, self._current_chat_history)
        self._emit_status_update(f"Switched to session.", "#98c379", True, 2000)

        if old_project_id != project_id:  # Re-check RAG status if project changed
            self._check_rag_readiness_and_emit_status()
        elif not self._is_rag_ready:  # Or if RAG wasn't ready before
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
        self._event_bus.requestRagScanDirectory.connect(self.request_rag_scan_directory)
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
        if backend_id == constants.GENERATOR_BACKEND_ID:
            actual_system_prompt = constants.CODER_AI_SYSTEM_PROMPT
            logger.info(f"CM: Using CODER_AI_SYSTEM_PROMPT for {backend_id}")
        if backend_id == "gemini_chat_default":
            api_key_to_use = get_gemini_api_key()
        elif backend_id == "gpt_chat_default":
            api_key_to_use = get_openai_api_key()
        elif backend_id in [constants.GENERATOR_BACKEND_ID, "ollama_chat_default"]:
            api_key_to_use = None
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

        if self._current_llm_request_id and processed_input.intent == UserInputIntent.NORMAL_CHAT:
            self._emit_status_update("Please wait for the current AI response.", "#e5c07b", True, 2500)
            return

        user_msg_parts = [user_msg_txt] if user_msg_txt else []
        if image_data: user_msg_parts.extend(image_data)
        user_message = ChatMessage(role=USER_ROLE, parts=user_msg_parts)
        self._current_chat_history.append(user_message)
        self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id, user_message)
        self._log_llm_comm("USER", user_msg_txt)

        if processed_input.intent == UserInputIntent.NORMAL_CHAT:
            if not self._is_chat_backend_configured:
                err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,
                                          parts=["[Error: Chat Backend not ready.]"])
                self._current_chat_history.append(err_msg_obj)
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              err_msg_obj)
                self._emit_status_update("Cannot send: Chat AI backend not configured.", "#FF6B6B")
                return

            history_for_llm = self._current_chat_history[:]
            rag_context_str = ""
            if self._rag_handler.should_perform_rag(user_msg_txt, self._is_rag_ready, self._is_rag_ready):
                logger.info(f"CM: Performing RAG for query: '{user_msg_txt[:50]}'")
                self._emit_status_update("Searching RAG context...", "#61afef", True, 1500)
                query_entities = self._rag_handler.extract_code_entities(user_msg_txt)
                rag_context_str, queried_collections = self._rag_handler.get_formatted_context(
                    query=user_msg_txt,
                    query_entities=query_entities,
                    project_id=self._current_project_id,  # Pass current project_id
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

            success, err, req_id = self._backend_coordinator.initiate_llm_chat_request(self._active_chat_backend_id,
                                                                                       history_for_llm, {
                                                                                           "temperature": self._active_temperature})
            if not success or not req_id:
                err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=[f"[Error sending: {err}]"])
                self._current_chat_history.append(err_msg_obj)
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              err_msg_obj)
                self._emit_status_update(f"Failed to start chat: {err or 'Unknown'}", "#FF6B6B")
                return
            self._current_llm_request_id = req_id
            placeholder = ChatMessage(id=req_id, role=MODEL_ROLE, parts=[""], loading_state=MessageLoadingState.LOADING)
            self._current_chat_history.append(placeholder)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          placeholder)
            self._emit_status_update(f"Sending to {self._active_chat_model_name}...", "#61afef")
            self._backend_coordinator.start_llm_streaming_task(req_id, self._active_chat_backend_id, history_for_llm,
                                                               False, {"temperature": self._active_temperature},
                                                               {"purpose": "normal_chat",
                                                                "project_id": self._current_project_id,
                                                                "session_id": self._current_session_id})
        elif processed_input.intent == UserInputIntent.PLAN_THEN_CODE_REQUEST:
            if not self._is_chat_backend_configured or not self._is_specialized_backend_configured:
                self._emit_status_update("Planner or Code LLM not configured.", "#e06c75", True, 5000)
                return
            self._plan_and_code_coordinator.start_planning_sequence(
                user_query=user_msg_txt,
                planner_llm_backend_id=self._active_chat_backend_id,
                planner_llm_model_name=self._active_chat_model_name,
                planner_llm_temperature=self._active_temperature,
                project_id=self._current_project_id, session_id=self._current_session_id,
                specialized_llm_backend_id=self._active_specialized_backend_id,
                specialized_llm_model_name=self._active_specialized_model_name
            )
        else:
            unknown_intent_msg = ChatMessage(role=ERROR_ROLE,
                                             parts=[f"[System: Unknown request type: {user_msg_txt[:50]}...]"])
            self._current_chat_history.append(unknown_intent_msg)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              unknown_intent_msg)

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

    @Slot(str, str)  # MODIFIED: Accept project_id from event
    def request_rag_scan_directory(self, dir_path: str, project_id: str):
        if not project_id:  # project_id must be valid for this operation
            logger.error("CM: RAG scan requested without a valid project_id.")
            self._emit_status_update("Cannot scan for RAG: Project ID missing.", "#e06c75", True, 3000)
            return

        # Check if the provided project_id matches the currently active one.
        # This specific implementation assumes the scan is for the *active* project.
        # If we wanted to allow scanning for non-active projects, this logic would change.
        if project_id != self._current_project_id:
            logger.warning(
                f"CM: RAG scan requested for project '{project_id}', but active project is '{self._current_project_id}'. Scan will target '{project_id}'.")
            # Potentially, we might want to prevent this or confirm with the user.
            # For now, we allow it, but it's an important consideration.
            # self._emit_status_update(f"Warning: Scanning for RAG for project '{project_id}', which is not active.", "#e5c07b", True, 5000)

        if not self._upload_service or not self._upload_service.is_vector_db_ready(
                project_id):  # Check readiness for specific project
            self._emit_status_update(f"RAG system not ready for project '{project_id}'. Cannot scan.", "#FF6B6B", True,
                                     4000)
            return

        logger.info(f"CM: Requesting RAG scan for directory: {dir_path} for project: {project_id}")
        self._emit_status_update(f"Scanning '{os.path.basename(dir_path)}' for RAG (Project: {project_id[:8]})...",
                                 "#61afef", False)

        # Pass the project_id as the collection_id to UploadService
        rag_message = self._upload_service.process_directory_for_context(dir_path, collection_id=project_id)

        if rag_message and self._current_project_id == project_id and self._current_session_id:  # Only add to current chat if it's for the active project
            self._current_chat_history.append(rag_message)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          rag_message)
            self._log_llm_comm(f"RAG_SCAN_P:{project_id[:6]}", rag_message.text)
            if rag_message.role == ERROR_ROLE:
                self._emit_status_update(f"RAG Scan for '{project_id[:8]}' completed with errors.", "#FF6B6B", True,
                                         5000)
            else:
                self._emit_status_update(f"RAG Scan for '{project_id[:8]}' complete!", "#98c379", True, 3000)
        elif rag_message:  # Scan was for a non-active project or no active session
            logger.info(
                f"RAG Scan for project '{project_id}' completed. Message (not added to active chat): {rag_message.text}")
            if rag_message.role == ERROR_ROLE:
                self._emit_status_update(f"RAG Scan for '{project_id[:8]}' (non-active) completed with errors.",
                                         "#FF6B6B", True, 5000)
            else:
                self._emit_status_update(f"RAG Scan for '{project_id[:8]}' (non-active) complete!", "#98c379", True,
                                         3000)
        else:
            self._emit_status_update(f"RAG Scan for '{project_id[:8]}' failed or returned no message.", "#FF6B6B", True,
                                     3000)

        self._check_rag_readiness_and_emit_status()

    @Slot(str, str)
    def _handle_llm_request_sent(self, backend_id: str, request_id: str):
        if request_id == self._current_llm_request_id and backend_id == self._active_chat_backend_id:
            self._event_bus.uiInputBarBusyStateChanged.emit(True)

    @Slot(str, object, dict)
    def _handle_llm_response_completed(self, request_id: str, completed_message_obj: object, usage_stats_dict: dict):
        meta_pid = usage_stats_dict.get("project_id")
        meta_sid = usage_stats_dict.get("session_id")
        if request_id == self._current_llm_request_id and meta_pid == self._current_project_id and meta_sid == self._current_session_id:
            if isinstance(completed_message_obj, ChatMessage):
                self._log_llm_comm(self._active_chat_backend_id.upper() + " RESPONSE", completed_message_obj.text)
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
            self._log_llm_comm(f"{self._active_chat_backend_id.upper()} ERROR", error_message_str)
            err_chat_msg = ChatMessage(id=request_id, role=ERROR_ROLE, parts=[f"[AI Error: {error_message_str}]"],
                                       loading_state=MessageLoadingState.ERROR)
            updated = False
            for i, msg in enumerate(self._current_chat_history):
                if msg.id == request_id: self._current_chat_history[i] = err_chat_msg; updated = True; break
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
                                                   '_vector_db_service') or not self._upload_service._vector_db_service:  # type: ignore
            self._is_rag_ready = False
            self._event_bus.ragStatusChanged.emit(False, "RAG Not Ready (Service Error)", "#e06c75")
            return

        rag_ready_for_project = False
        rag_status_message = "RAG Status: Initializing..."
        rag_status_color = constants.TIMESTAMP_COLOR_HEX  # Muted color

        if not self._current_project_id:
            rag_status_message = "RAG Status: No Active Project"
            self._is_rag_ready = False  # Overall RAG readiness is tied to active project for project-specific RAG
        elif not self._upload_service.is_vector_db_ready(self._current_project_id):  # Check specific project
            rag_status_message = f"RAG Not Ready for '{self._current_project_id[:8]}...' (DB Error)"
            rag_status_color = "#e06c75"  # Red
            self._is_rag_ready = False
        elif not self._upload_service._dependencies_ready:  # type: ignore
            rag_status_message = "RAG Not Ready (Dependencies Missing)"
            rag_status_color = "#e06c75"  # Red
            self._is_rag_ready = False
        else:
            project_collection_size = self._upload_service._vector_db_service.get_collection_size(
                self._current_project_id)  # type: ignore
            if project_collection_size == -1:  # Error fetching size
                rag_status_message = f"RAG Status for '{self._current_project_id[:8]}...': DB Error"
                rag_status_color = "#e06c75"  # Red
                self._is_rag_ready = False  # If we can't get size, assume not fully ready
            elif project_collection_size == 0:
                rag_status_message = f"RAG Ready for '{self._current_project_id[:8]}...' (Empty)"
                rag_status_color = "#e5c07b"  # Yellow for ready but empty
                rag_ready_for_project = True
                self._is_rag_ready = True
            else:
                rag_status_message = f"RAG Ready for '{self._current_project_id[:8]}...' ({project_collection_size} chunks)"
                rag_status_color = "#98c379"  # Green
                rag_ready_for_project = True
                self._is_rag_ready = True

        logger.info(
            f"CM: Emitting RAG Status: Ready={self._is_rag_ready}, ProjectReady={rag_ready_for_project}, Msg='{rag_status_message}'")
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
        # This now reflects readiness for the *active project* or "no active project"
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