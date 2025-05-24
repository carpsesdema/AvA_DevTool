import logging
import uuid
from typing import List, Optional, Dict, Any

from PySide6.QtCore import QObject, Slot

try:
    from core.application_orchestrator import ApplicationOrchestrator
    from core.event_bus import EventBus
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from core.message_enums import MessageLoadingState
    from backends.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from utils import constants
    from config import get_gemini_api_key, get_openai_api_key  # ADDED get_openai_api_key
except ImportError as e:
    logging.getLogger(__name__).critical(f"Critical import error in ChatManager: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatManager(QObject):
    def __init__(self, orchestrator: ApplicationOrchestrator, parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ChatManager initializing (Phase 1)...")

        if not isinstance(orchestrator, ApplicationOrchestrator):
            logger.critical("ChatManager requires a valid ApplicationOrchestrator.")
            raise TypeError("ChatManager requires a valid ApplicationOrchestrator.")

        self._orchestrator = orchestrator
        self._event_bus = orchestrator.get_event_bus()
        self._backend_coordinator = orchestrator.get_backend_coordinator()
        self._llm_comm_logger = orchestrator.get_llm_communication_logger()

        self._current_chat_history: List[ChatMessage] = []
        self._active_chat_backend_id: str = constants.DEFAULT_CHAT_BACKEND_ID
        self._active_model_name: str = (
            constants.DEFAULT_OLLAMA_CHAT_MODEL  # CORRECTED: Use Ollama default if backend is Ollama
            if constants.DEFAULT_CHAT_BACKEND_ID == "ollama_chat_default"
            else constants.DEFAULT_GEMINI_CHAT_MODEL
        )
        self._active_personality_prompt: Optional[
            str] = "You are Ava, a bubbly, enthusiastic, and incredibly helpful AI assistant!"
        self._active_temperature: float = 0.7
        self._is_current_backend_configured: bool = False
        self._current_llm_request_id: Optional[str] = None

        self._connect_event_bus_subscriptions()
        logger.info("ChatManager (Phase 1) initialized and subscriptions connected.")

    def initialize(self):
        logger.info("ChatManager late initialization (Phase 1)...")
        # This is where we trigger the initial configuration of the backend
        # so the UI gets configured status and model list.
        self._configure_active_chat_backend()

    def _connect_event_bus_subscriptions(self):
        logger.debug("ChatManager connecting EventBus subscriptions...")
        self._event_bus.userMessageSubmitted.connect(self.handle_user_message)
        self._event_bus.newChatRequested.connect(self.start_new_chat_session)
        self._event_bus.chatLlmPersonalitySubmitted.connect(self._handle_personality_change_event)

        self._event_bus.llmRequestSent.connect(self._handle_llm_request_sent)
        self._event_bus.llmStreamChunkReceived.connect(self._handle_llm_stream_chunk)
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_response_completed)
        self._event_bus.llmResponseError.connect(self._handle_llm_response_error)
        # CORRECTED: This signal will update the UI via LeftPanel, but ChatManager also needs to listen
        # to ensure its internal state (_is_current_backend_configured) is up-to-date based on the actual result
        # of the configuration call.
        self._event_bus.backendConfigurationChanged.connect(self._handle_backend_reconfigured_event)
        logger.debug("ChatManager EventBus subscriptions connected.")

    def _emit_status_update(self, message: str, color: str, is_temporary: bool = False, duration_ms: int = 0):
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, is_temporary, duration_ms)

    def _log_llm_comm(self, sender: str, message: str):
        if self._llm_comm_logger:
            self._llm_comm_logger.log_message(sender, message)
        else:
            logger.info(f"LLM_COMM_LOG_FALLBACK: [{sender}] {message[:150]}...")

    def _configure_active_chat_backend(self):
        logger.info(f"CM: Configuring backend '{self._active_chat_backend_id}' with model '{self._active_model_name}'")
        api_key_to_use: Optional[str] = None

        # Logic to get the correct API key based on the selected backend
        if self._active_chat_backend_id == "gemini_chat_default":
            api_key_to_use = get_gemini_api_key()
            if not api_key_to_use:
                err_msg = "Gemini API Key not found. Cannot configure. Set GEMINI_API_KEY in .env"
                logger.error(err_msg)
                self._is_current_backend_configured = False
                # Emit to update UI about failed configuration
                self._event_bus.backendConfigurationChanged.emit(self._active_chat_backend_id, self._active_model_name,
                                                                 False, [])
                self._emit_status_update(err_msg, "#FF6B6B")
                return  # Stop configuration if key is missing
        elif self._active_chat_backend_id == "ollama_chat_default":
            api_key_to_use = None  # Ollama typically doesn't use API keys directly, connection status handled by adapter
        elif self._active_chat_backend_id == "gpt_chat_default":
            api_key_to_use = get_openai_api_key()
            if not api_key_to_use:
                err_msg = "OpenAI API Key not found. Cannot configure. Set OPENAI_API_KEY in .env"
                logger.error(err_msg)
                self._is_current_backend_configured = False
                # Emit to update UI about failed configuration
                self._event_bus.backendConfigurationChanged.emit(self._active_chat_backend_id, self._active_model_name,
                                                                 False, [])
                self._emit_status_update(err_msg, "#FF6B6B")
                return  # Stop configuration if key is missing

        # Attempt to configure the backend via the coordinator
        self._backend_coordinator.configure_backend(
            backend_id=self._active_chat_backend_id,
            api_key=api_key_to_use,
            model_name=self._active_model_name,
            system_prompt=self._active_personality_prompt
        )

    @Slot(str, str, bool, list)
    def _handle_backend_reconfigured_event(self, backend_id: str, model_name: str, is_configured: bool,
                                           available_models: list):
        # This slot is connected to EventBus.backendConfigurationChanged.
        # It's primarily for ChatManager to update its internal _is_current_backend_configured state
        # and to update the status bar if the configured backend is the currently active one.
        if backend_id == self._active_chat_backend_id and model_name == self._active_model_name:
            self._is_current_backend_configured = is_configured
            if is_configured:
                logger.info(f"CM: Backend '{backend_id}' for model '{model_name}' configured.")
                self._emit_status_update(f"Ready. Using {self._active_model_name}", "#98c379", False)
            else:
                last_error = self._backend_coordinator.get_last_error_for_backend(backend_id)
                err_msg = f"Failed to configure {backend_id} ({model_name}): {last_error or 'Unknown'}"
                logger.error(err_msg)
                self._emit_status_update(err_msg, "#FF6B6B")
        # Note: The LeftPanel also listens to this signal to update its combobox.

    @Slot(str, list)
    def handle_user_message(self, text: str, image_data: List[Dict[str, Any]]):
        logger.info(f"CM: User message: '{text[:50]}...'")
        text_stripped = text.strip()
        if not text_stripped:
            return

        if self._current_llm_request_id:
            self._emit_status_update("Please wait for the current AI response.", "#e5c07b", True, 2500)
            return

        if not self._is_current_backend_configured:
            err_msg = "Cannot send: AI backend not configured."
            logger.error(err_msg)
            self._emit_status_update(err_msg, "#FF6B6B")
            # CORRECTED: Use ERROR_ROLE for messages generated due to internal errors.
            error_chat_msg = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,  # Assign a new ID for the error message
                                         parts=[f"[Error: Backend not ready. Your message: '{text_stripped}']"])
            self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", error_chat_msg)
            return

        # Create user message and add to history/UI
        user_message = ChatMessage(role=USER_ROLE, parts=[text_stripped])
        self._current_chat_history.append(user_message)
        self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", user_message)
        self._log_llm_comm("USER", text_stripped)

        history_for_llm = self._current_chat_history[:]
        init_success, init_error_msg, assigned_request_id = self._backend_coordinator.initiate_llm_chat_request(
            target_backend_id=self._active_chat_backend_id,
            history_to_send=history_for_llm,
            options={"temperature": self._active_temperature}
        )

        if not init_success or not assigned_request_id:
            err_ui = f"Failed to start chat: {init_error_msg or 'Unknown'}"
            logger.error(f"CM: BC.initiate_llm_chat_request FAILED. {err_ui}")
            self._emit_status_update(err_ui, "#FF6B6B")
            error_chat_msg = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE,
                                         parts=[f"[Error sending: {init_error_msg}]"])
            self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", error_chat_msg)
            return

        self._current_llm_request_id = assigned_request_id
        # Add AI placeholder message to history/UI with LOADING state
        ai_placeholder_msg = ChatMessage(id=self._current_llm_request_id, role=MODEL_ROLE, parts=[""],
                                         loading_state=MessageLoadingState.LOADING)
        self._current_chat_history.append(ai_placeholder_msg)
        self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", ai_placeholder_msg)
        self._emit_status_update(f"Sending to {self._active_model_name}...", "#61afef")

        # Start the actual streaming task
        self._backend_coordinator.start_llm_streaming_task(
            request_id=self._current_llm_request_id,
            target_backend_id=self._active_chat_backend_id,
            history_to_send=history_for_llm,
            is_modification_response_expected=False,
            options={"temperature": self._active_temperature},
            request_metadata={"purpose": "p1_normal_chat", "user_query_start": text_stripped[:30], "project_id": "p1_chat_context"}  # FIXED: Add project_id
        )

    @Slot()
    def start_new_chat_session(self):
        logger.info("CM: New chat session requested.")
        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(request_id=self._current_llm_request_id)
            self._current_llm_request_id = None  # CORRECTED: Clear immediately on new chat
            self._event_bus.uiInputBarBusyStateChanged.emit(
                False)  # CORRECTED: Ensure UI is not busy after cancellation

        self._current_chat_history = []
        self._event_bus.activeSessionHistoryCleared.emit("p1_chat_context")
        self._emit_status_update("New chat started.", "#98c379", True, 2000)

    @Slot(str, str)
    def _handle_llm_request_sent(self, backend_id: str, request_id: str):
        if request_id == self._current_llm_request_id:
            self._event_bus.uiInputBarBusyStateChanged.emit(True)

    @Slot(str, str)
    def _handle_llm_stream_chunk(self, request_id: str, chunk_text: str):
        # FIXED: Actually handle the chunk by logging it
        if request_id == self._current_llm_request_id:
            logger.debug(f"CM: Received chunk for {request_id}: '{chunk_text[:50]}...'")

    @Slot(str, object, dict)  # FIXED: Changed ChatMessage to object to match signal
    def _handle_llm_response_completed(self, request_id: str, completed_message_obj: object,
                                       usage_stats_dict: dict):
        if request_id == self._current_llm_request_id:
            # Ensure we have a proper ChatMessage object
            if isinstance(completed_message_obj, ChatMessage):
                self._log_llm_comm(self._active_chat_backend_id.upper() + " RESPONSE", completed_message_obj.text)

                # Update the message in the internal history list
                updated_in_history = False
                for i, msg in enumerate(self._current_chat_history):
                    if msg.id == request_id:
                        self._current_chat_history[i] = completed_message_obj  # Replace placeholder with final message
                        # The completed_message_obj should already have the COMPLETED state from BackendCoordinator,
                        # but we ensure it here for consistency if necessary.
                        self._current_chat_history[i].loading_state = MessageLoadingState.COMPLETED
                        updated_in_history = True
                        break
                if not updated_in_history:
                    # Fallback: If message was somehow not found, append it (shouldn't happen with correct flow)
                    self._current_chat_history.append(completed_message_obj)

            self._current_llm_request_id = None  # Clear active request ID
            self._event_bus.uiInputBarBusyStateChanged.emit(False)  # Release UI input bar
            self._emit_status_update(f"Ready. Last: {self._active_model_name}", "#98c379")

    @Slot(str, str)
    def _handle_llm_response_error(self, request_id: str, error_message_str: str):
        if request_id == self._current_llm_request_id:
            self._log_llm_comm(f"{self._active_chat_backend_id.upper()} ERROR", error_message_str)

            # Create an error message object for display
            error_chat_msg = ChatMessage(id=request_id, role=ERROR_ROLE, parts=[f"[AI Error: {error_message_str}]"],
                                         loading_state=MessageLoadingState.ERROR)

            # Find and replace the placeholder message in internal history
            updated_in_history = False
            for i, msg in enumerate(self._current_chat_history):
                if msg.id == request_id:
                    self._current_chat_history[i] = error_chat_msg
                    updated_in_history = True
                    break
            if not updated_in_history:
                # Fallback: If message was not found, append the error message
                self._current_chat_history.append(error_chat_msg)

            self._current_llm_request_id = None  # Clear active request ID
            self._event_bus.uiInputBarBusyStateChanged.emit(False)  # Release UI input bar
            self._emit_status_update(f"Error: {error_message_str[:60]}...", "#FF6B6B")

    def set_active_chat_model_and_backend(self, backend_id: str, model_name: str):
        if not backend_id or not model_name: return
        # Only reconfigure if model or backend actually changed to avoid unnecessary re-initialization
        if self._active_chat_backend_id != backend_id or self._active_model_name != model_name:
            self._active_chat_backend_id = backend_id
            self._active_model_name = model_name
            self._configure_active_chat_backend()
        else:
            logger.debug(
                f"CM: Model '{model_name}' for backend '{backend_id}' already active. No reconfiguration needed.")

    @Slot(str, str)
    def _handle_personality_change_event(self, new_prompt: str, backend_id_for_persona: str):
        if backend_id_for_persona == self._active_chat_backend_id:
            self._active_personality_prompt = new_prompt.strip() if new_prompt and new_prompt.strip() else None
            self._configure_active_chat_backend()  # Reconfigure with new personality to apply prompt

    def set_chat_temperature(self, temperature: float):
        if 0.0 <= temperature <= 2.0:
            self._active_temperature = temperature
            self._emit_status_update(f"Temperature: {temperature:.2f}", "#61afef", True, 1500)

    def get_current_chat_history(self, project_id_placeholder: str = "p1_chat_context") -> List[ChatMessage]:
        return self._current_chat_history

    def get_current_active_chat_backend_id(self) -> str:
        return self._active_chat_backend_id

    def get_model_for_backend(self, backend_id: str) -> Optional[str]:
        # Returns the *currently configured* model name for a specific backend ID
        return self._backend_coordinator.get_current_configured_model(backend_id)

    def get_all_available_backend_ids(self) -> List[str]:
        """Returns a list of all backend IDs managed by the coordinator."""
        return self._backend_coordinator.get_all_backend_ids()

    def get_available_models_for_backend(self, backend_id: str) -> List[str]:
        """Returns a list of available models for a specific backend."""
        return self._backend_coordinator.get_available_models_for_backend(backend_id)

    def get_current_chat_personality(self) -> Optional[str]:
        return self._active_personality_prompt

    def get_current_chat_temperature(self) -> float:
        return self._active_temperature

    def is_api_ready(self) -> bool:
        # Check if the currently active backend is configured
        return self._is_current_backend_configured

    def is_overall_busy(self) -> bool:
        # Check if a chat request is active OR if the backend coordinator reports overall busy (e.g., fetching models)
        return bool(self._current_llm_request_id) or self._backend_coordinator.is_any_backend_busy()

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:
        return self._llm_comm_logger

    def get_backend_coordinator(self) -> BackendCoordinator:
        return self._backend_coordinator

    def cleanup_phase1(self):
        logger.info("ChatManager (Phase 1) cleanup...")
        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(request_id=self._current_llm_request_id)