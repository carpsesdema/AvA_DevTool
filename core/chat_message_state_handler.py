# core/chat_message_state_handler.py
import logging
from typing import Optional, Dict, List, Tuple  # ADDED List

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from ui.chat_list_model import ChatListModel  # Keep this for direct model interaction
    from core.message_enums import MessageLoadingState
    from core.models import ChatMessage
except ImportError as e_cmsh:
    logging.getLogger(__name__).critical(f"ChatMessageStateHandler: Critical import error: {e_cmsh}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatMessageStateHandler(QObject):
    def __init__(self,
                 event_bus: EventBus,
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(event_bus, EventBus):
            logger.critical("ChatMessageStateHandler requires a valid EventBus instance.")
            raise TypeError("ChatMessageStateHandler requires a valid EventBus instance.")

        self._event_bus = event_bus
        # MODIFIED: Store models per project_id AND session_id
        self._active_models: Dict[Tuple[str, str], ChatListModel] = {}  # (project_id, session_id): model

        self._connect_event_bus_subscriptions()
        logger.info("ChatMessageStateHandler initialized and connected to EventBus LLM signals.")

    def _connect_event_bus_subscriptions(self):
        # These global LLM signals might still be useful if they carry enough context (like request_id which can be mapped)
        # However, specific per-session signals are more robust.
        self._event_bus.llmStreamStarted.connect(
            self._handle_llm_stream_started)  # Potentially keep if request_id is unique across sessions

        # MODIFIED: Listen to per-session finalization/error signals
        self._event_bus.messageFinalizedForSession.connect(self._handle_llm_response_completed_for_session)
        # self._event_bus.llmResponseError.connect(self._handle_llm_response_error) # Replaced by messageFinalizedForSession

        # Listen to when active session's history is loaded to potentially (re)register its model
        self._event_bus.activeSessionHistoryLoaded.connect(self._handle_active_session_history_loaded)
        self._event_bus.activeSessionHistoryCleared.connect(self._handle_active_session_cleared)

    # MODIFIED: Register model for a specific project and session
    def register_model_for_project_session(self, project_id: str, session_id: str, model: ChatListModel):
        if not all(isinstance(arg, str) and arg.strip() for arg in [project_id, session_id]):
            logger.warning("CMSH: Attempted to register model with invalid project_id or session_id.")
            return
        if not isinstance(model, ChatListModel):
            logger.warning(
                f"CMSH: Attempted to register invalid model type for project '{project_id}', session '{session_id}'.")
            return

        key = (project_id, session_id)
        logger.info(f"CMSH: Registering ChatListModel for Project/Session: '{project_id}/{session_id}'.")
        self._active_models[key] = model

    # MODIFIED: Unregister model for a specific project and session
    def unregister_model_for_project_session(self, project_id: str, session_id: str):
        key = (project_id, session_id)
        if key in self._active_models:
            logger.info(f"CMSH: Unregistering ChatListModel for Project/Session: '{project_id}/{session_id}'.")
            del self._active_models[key]
        else:
            logger.debug(f"CMSH: No model found to unregister for Project/Session: '{project_id}/{session_id}'.")

    # MODIFIED: Get model based on project and session IDs from metadata
    def _get_model_for_request_context(self, project_id: Optional[str], session_id: Optional[str]) -> Optional[
        ChatListModel]:
        if project_id and session_id:
            key = (project_id, session_id)
            return self._active_models.get(key)
        logger.warning(
            f"CMSH: Could not find a registered ChatListModel for Project/Session: '{project_id}/{session_id}'. Active models: {list(self._active_models.keys())}")
        return None

    @Slot(str, str, list)  # ADDED: project_id, session_id, history_list
    def _handle_active_session_history_loaded(self, project_id: str, session_id: str, history: List[ChatMessage]):
        # This is where the MainWindow should inform CMSH about the model for the *active* display area.
        # Assuming MainWindow has a way to get its currently active ChatDisplayArea's model.
        # For now, this method doesn't automatically register. Registration should be explicit.
        logger.debug(
            f"CMSH: Noted active session history loaded for P:{project_id} S:{session_id}. CMSH awaits model registration if needed.")

    @Slot(str, str)  # ADDED
    def _handle_active_session_cleared(self, project_id: str, session_id: str):
        # If a model was specifically tied to this project/session and needs cleanup, do it here.
        # Or, the UI component owning the model might unregister it.
        logger.debug(f"CMSH: Noted active session cleared for P:{project_id} S:{session_id}.")

    @Slot(str)  # request_id
    def _handle_llm_stream_started(self, request_id: str):
        logger.debug(f"CMSH Event: llmStreamStarted for ReqID '{request_id}'.")
        # This is tricky if request_id isn't globally unique OR if we don't know which session it belongs to.
        # The ChatManager needs to associate request_id with a session and inform CMSH,
        # or the llmStreamStarted signal needs to include session_id.
        # For now, iterate through all models to find the message. This is inefficient.
        model_to_update = None
        found_in_context = "unknown"
        for (pid, sid), model_instance in self._active_models.items():
            if model_instance.find_message_row_by_id(request_id) is not None:
                model_to_update = model_instance
                found_in_context = f"P:{pid}/S:{sid}"
                logger.debug(
                    f"CMSH: Found model for context '{found_in_context}' containing message for ReqID '{request_id}'.")
                break

        if model_to_update:
            model_to_update.update_message_loading_state_by_id(request_id, MessageLoadingState.LOADING)
        else:
            logger.warning(
                f"CMSH: No registered model found containing message for ReqID '{request_id}' to mark as LOADING.")

    # MODIFIED: Handles the new messageFinalizedForSession signal
    @Slot(str, str, str, ChatMessage, dict, bool)
    def _handle_llm_response_completed_for_session(self,
                                                   project_id: str,
                                                   session_id: str,
                                                   request_id: str,
                                                   completed_message: ChatMessage,
                                                   usage_stats_with_metadata: dict,
                                                   is_error: bool):
        logger.debug(
            f"CMSH Event: messageFinalizedForSession for P:{project_id} S:{session_id} ReqID '{request_id}'. IsError: {is_error}")
        model_to_update = self._get_model_for_request_context(project_id, session_id)

        if model_to_update:
            new_state = MessageLoadingState.ERROR if is_error else MessageLoadingState.COMPLETED
            # The ChatListModel's finalize_message_by_id (or a similar method) should handle content update.
            # Here, we ensure its loading state is set.
            # If the model's `finalize_message_by_id` also sets loading state, this might be redundant,
            # but it's safer to ensure it here as CMSH's direct responsibility.
            success_state_update = model_to_update.update_message_loading_state_by_id(request_id, new_state)
            if not success_state_update:
                logger.warning(
                    f"CMSH: Failed to find message with ID '{request_id}' in its model (P:{project_id} S:{session_id}) to mark as {new_state.name}.")
            # The ChatListModel itself should be listening to messageFinalizedForSession to update content if needed,
            # or ChatManager ensures the ChatMessage object passed already has the final content.
        else:
            logger.warning(
                f"CMSH: No model found for P:{project_id} S:{session_id} (ReqID '{request_id}') to mark as {'ERROR' if is_error else 'COMPLETED'}.")