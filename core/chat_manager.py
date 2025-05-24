# core/chat_manager.py
import logging
import uuid
from typing import List, Optional, Dict, Any, TYPE_CHECKING  # ADDED TYPE_CHECKING

from PySide6.QtCore import QObject, Slot

# Conditional import to break circular dependency for type hinting
if TYPE_CHECKING:
    from core.application_orchestrator import ApplicationOrchestrator

try:
    # from core.application_orchestrator import ApplicationOrchestrator # This line is the problem
    from core.event_bus import EventBus
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from core.message_enums import MessageLoadingState
    from backends.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from services.project_service import ProjectManager  # type: ignore
    from utils import constants
    from config import get_gemini_api_key, get_openai_api_key
    from core.user_input_handler import UserInputHandler, UserInputIntent
    from core.plan_and_code_coordinator import PlanAndCodeCoordinator
except ImportError as e:
    ProjectManager = type("ProjectManager", (object,), {})  # type: ignore
    # ApplicationOrchestrator = type("ApplicationOrchestrator", (object,), {}) # type: ignore # Not needed if string literal used
    logging.getLogger(__name__).critical(f"Critical import error in ChatManager: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatManager(QObject):
    def __init__(self, orchestrator: 'ApplicationOrchestrator', parent: Optional[QObject] = None):  # Use string literal
        super().__init__(parent)
        logger.info("ChatManager initializing...")

        # Ensure orchestrator is an instance of the actual class at runtime, even if hinted as string
        # This check might be problematic if 'ApplicationOrchestrator' itself isn't fully defined due to the fix.
        # We might need to rely on duck typing or a less strict check if 'core.application_orchestrator'
        # isn't imported directly here anymore. For now, let's assume it's okay or handled by main.py's setup.
        # from core.application_orchestrator import ApplicationOrchestrator as AppOrchReal # Local import for runtime check
        # if not isinstance(orchestrator, AppOrchReal):
        #     logger.critical("ChatManager requires a valid ApplicationOrchestrator.")
        #     raise TypeError("ChatManager requires a valid ApplicationOrchestrator.")
        self._orchestrator = orchestrator

        self._event_bus = self._orchestrator.get_event_bus()
        self._backend_coordinator = self._orchestrator.get_backend_coordinator()
        self._llm_comm_logger = self._orchestrator.get_llm_communication_logger()

        self._project_manager = self._orchestrator.get_project_manager()
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
        self._current_llm_request_id: Optional[str] = None

        self._connect_event_bus_subscriptions()
        logger.info("ChatManager initialized and subscriptions connected.")

    def initialize(self):
        logger.info("ChatManager late initialization...")
        self._configure_backend(self._active_chat_backend_id, self._active_chat_model_name,
                                self._active_chat_personality_prompt)
        self._configure_backend(self._active_specialized_backend_id, self._active_specialized_model_name,
                                constants.CODER_AI_SYSTEM_PROMPT)

    def set_active_session(self, project_id: str, session_id: str, history: List[ChatMessage]):
        logger.info(f"CM: Setting active session to P:{project_id}/S:{session_id}. History items: {len(history)}")
        self._current_project_id = project_id
        self._current_session_id = session_id
        self._current_chat_history = list(history)

        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(self._current_llm_request_id)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)

        self._event_bus.activeSessionHistoryCleared.emit(project_id, session_id)
        self._event_bus.activeSessionHistoryLoaded.emit(project_id, session_id, self._current_chat_history)
        self._emit_status_update(f"Switched to session.", "#98c379", True,
                                 2000)  # Removed P/S details, MainWindow title handles it

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
                user_query=user_msg_txt,  # Use user_msg_txt which is stripped
                planner_llm_backend_id=self._active_chat_backend_id,
                planner_llm_model_name=self._active_chat_model_name,
                planner_llm_temperature=self._active_temperature,
                project_id=self._current_project_id, session_id=self._current_session_id,
                specialized_llm_backend_id=self._active_specialized_backend_id,
                specialized_llm_model_name=self._active_specialized_model_name
            )
        else:  # Unknown intent
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
                if self._current_project_id and self._current_session_id:  # Ensure context before emitting
                    self._event_bus.messageFinalizedForSession.emit(self._current_project_id, self._current_session_id,
                                                                    request_id, completed_message_obj, usage_stats_dict,
                                                                    False)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._emit_status_update(f"Ready. Last: {self._active_chat_model_name}", "#98c379")

    @Slot(str, str)
    def _handle_llm_response_error(self, request_id: str, error_message_str: str):
        # Assuming error is for current P/S if request_id matches
        if request_id == self._current_llm_request_id:
            self._log_llm_comm(f"{self._active_chat_backend_id.upper()} ERROR", error_message_str)
            err_chat_msg = ChatMessage(id=request_id, role=ERROR_ROLE, parts=[f"[AI Error: {error_message_str}]"],
                                       loading_state=MessageLoadingState.ERROR)
            updated = False
            for i, msg in enumerate(self._current_chat_history):
                if msg.id == request_id: self._current_chat_history[i] = err_chat_msg; updated = True; break
            if not updated: self._current_chat_history.append(err_chat_msg)
            if self._current_project_id and self._current_session_id:  # Ensure context
                self._event_bus.messageFinalizedForSession.emit(self._current_project_id, self._current_session_id,
                                                                request_id, err_chat_msg, {}, True)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._emit_status_update(f"Error: {error_message_str[:60]}...", "#FF6B6B")

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

    def set_model_for_backend(self, backend_id: str, model_name: str):  # Public method
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

    def is_overall_busy(self) -> bool:
        return bool(self._current_llm_request_id) or self._backend_coordinator.is_any_backend_busy()

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:
        return self._llm_comm_logger

    def get_backend_coordinator(self) -> BackendCoordinator:
        return self._backend_coordinator

    def get_project_manager(self) -> ProjectManager:
        return self._project_manager  # type: ignore

    def cleanup_phase1(self):  # Renamed to cleanup
        logger.info("ChatManager cleanup...")
        if self._current_llm_request_id: self._backend_coordinator.cancel_current_task(
            self._current_llm_request_id); self._current_llm_request_id = None