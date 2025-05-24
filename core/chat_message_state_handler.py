import logging
from typing import Optional, Dict

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from ui.chat_list_model import ChatListModel
    from core.message_enums import MessageLoadingState
    from core.models import ChatMessage
except ImportError as e_cmsh:
    logging.getLogger(__name__).critical(f"ChatMessageStateHandler: Critical import error: {e_cmsh}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatMessageStateHandler(QObject):
    def __init__(self,
                 event_bus: EventBus,  # Expect EventBus to be passed
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        if not isinstance(event_bus, EventBus):
            logger.critical("ChatMessageStateHandler requires a valid EventBus instance.")
            raise TypeError("ChatMessageStateHandler requires a valid EventBus instance.")

        self._event_bus = event_bus
        self._active_models: Dict[str, ChatListModel] = {}  # project_id: model

        self._connect_event_bus_subscriptions()
        logger.info("ChatMessageStateHandler initialized and connected to EventBus LLM signals.")

    def _connect_event_bus_subscriptions(self):
        self._event_bus.llmStreamStarted.connect(self._handle_llm_stream_started)
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_response_completed)
        self._event_bus.llmResponseError.connect(self._handle_llm_response_error)

        self._event_bus.activeProjectChanged.connect(self._handle_active_project_changed)

    def register_model_for_project(self, project_id: str, model: ChatListModel):
        if not isinstance(project_id, str) or not project_id.strip():
            logger.warning("CMSH: Attempted to register model with invalid project_id.")
            return
        if not isinstance(model, ChatListModel):
            logger.warning(f"CMSH: Attempted to register invalid model type for project '{project_id}'.")
            return

        logger.info(f"CMSH: Registering ChatListModel for project_id '{project_id}'.")
        self._active_models[project_id] = model

    def unregister_model_for_project(self, project_id: str):
        if project_id in self._active_models:
            logger.info(f"CMSH: Unregistering ChatListModel for project_id '{project_id}'.")
            del self._active_models[project_id]
        else:
            logger.debug(f"CMSH: No model found to unregister for project_id '{project_id}'.")

    def _get_model_for_request(self, request_id: str, usage_stats_or_metadata: Optional[Dict] = None) -> Optional[
        ChatListModel]:
        project_id_from_meta = None
        if usage_stats_or_metadata and isinstance(usage_stats_or_metadata, dict):
            project_id_from_meta = usage_stats_or_metadata.get("project_id")
            if not project_id_from_meta:  # Fallback for older metadata structures
                project_id_from_meta = usage_stats_or_metadata.get("p1_chat_context")  # P1 used this

        if project_id_from_meta and project_id_from_meta in self._active_models:
            return self._active_models[project_id_from_meta]

        logger.warning(
            f"CMSH: Could not find a registered ChatListModel for request_id '{request_id}' (Project ID from meta: '{project_id_from_meta}'). Active models: {list(self._active_models.keys())}")
        return None

    @Slot(str)
    def _handle_active_project_changed(self, project_id: str):
        pass

    @Slot(str)
    def _handle_llm_stream_started(self, request_id: str):
        logger.debug(f"CMSH Event: llmStreamStarted for ReqID '{request_id}'.")

        model_to_update = None
        for pid, model_instance in self._active_models.items():
            if model_instance.find_message_row_by_id(request_id) is not None:
                model_to_update = model_instance
                logger.debug(f"CMSH: Found model for project '{pid}' containing message for ReqID '{request_id}'.")
                break

        if model_to_update:
            model_to_update.update_message_loading_state_by_id(request_id, MessageLoadingState.LOADING)
        else:
            logger.warning(
                f"CMSH: No registered model found containing message for ReqID '{request_id}' to mark as LOADING.")

    @Slot(str, ChatMessage, dict)
    def _handle_llm_response_completed(self, request_id: str, completed_message: ChatMessage,
                                       usage_stats_with_metadata: dict):
        logger.debug(f"CMSH Event: llmResponseCompleted for ReqID '{request_id}'.")
        model_to_update = self._get_model_for_request(request_id, usage_stats_with_metadata)

        if model_to_update:
            success = model_to_update.update_message_loading_state_by_id(request_id, MessageLoadingState.COMPLETED)
            if not success:
                logger.warning(
                    f"CMSH: Failed to find message with ID '{request_id}' in its model to mark as COMPLETED.")
        else:
            logger.warning(
                f"CMSH: No model found for ReqID '{request_id}' to mark as COMPLETED. Usage/Meta: {usage_stats_with_metadata}")

    @Slot(str, str)
    def _handle_llm_response_error(self, request_id: str, error_message_str: str):
        logger.debug(f"CMSH Event: llmResponseError for ReqID '{request_id}'.")

        model_to_update = None
        error_origin_project_id = "unknown_project"

        for pid, model_instance in self._active_models.items():
            if model_instance.find_message_row_by_id(request_id) is not None:
                model_to_update = model_instance
                error_origin_project_id = pid
                logger.debug(
                    f"CMSH: Found model for project '{pid}' containing message for errored ReqID '{request_id}'.")
                break

        if model_to_update:
            success = model_to_update.update_message_loading_state_by_id(request_id, MessageLoadingState.ERROR)
            if not success:
                logger.warning(
                    f"CMSH: Failed to find message ID '{request_id}' in project '{error_origin_project_id}' model to mark as ERROR.")
        else:
            logger.warning(f"CMSH: No registered model found containing message for errored ReqID '{request_id}'.")

