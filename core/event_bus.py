# core/event_bus.py
import logging
from typing import Optional
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class EventBus(QObject):
    _instance: Optional['EventBus'] = None

    # User Actions
    userMessageSubmitted = Signal(str, list)  # text, image_data_list
    newChatRequested = Signal()  # UI requests CM to signal orchestrator for new session
    chatLlmPersonalitySubmitted = Signal(str, str)  # new_prompt, backend_id_for_persona
    chatLlmSelectionChanged = Signal(str, str)  # backend_id, model_name
    specializedLlmSelectionChanged = Signal(str, str)  # backend_id, model_name

    # RAG Signals
    # MODIFICATION: Confirmed requestRagScanDirectory is for GLOBAL RAG and takes only dir_path
    requestRagScanDirectory = Signal(str)  # dir_path (targets GLOBAL RAG)
    # NEW: Signal for project-specific file uploads
    requestProjectFilesUpload = Signal(list, str)  # file_paths: List[str], project_id: str

    # Orchestrator-level request for new session
    createNewSessionForProjectRequested = Signal(str)  # project_id (ChatManager emits this)

    # UI Navigation / Dialog Triggers
    showLlmLogWindowRequested = Signal()
    chatLlmPersonalityEditRequested = Signal()
    viewCodeViewerRequested = Signal()
    createNewProjectRequested = Signal(str, str) # project_name, project_description
    openProjectSelectorRequested = Signal()
    renameCurrentSessionRequested = Signal(str) # new_session_name

    # Backend & LLM Communication
    backendConfigurationChanged = Signal(str, str, bool, list) # backend_id, model_name, is_configured, available_models
    llmRequestSent = Signal(str, str) # backend_id, request_id
    llmStreamStarted = Signal(str) # request_id
    llmStreamChunkReceived = Signal(str, str) # request_id, chunk_text
    llmResponseCompleted = Signal(str, object, dict) # request_id, chat_message_obj, usage_stats_dict
    llmResponseError = Signal(str, str) # request_id, error_message_str

    # Chat History & Session Management (MODIFIED to include project/session IDs)
    newMessageAddedToHistory = Signal(str, str, object)  # project_id, session_id, chat_message_obj
    activeSessionHistoryCleared = Signal(str, str)  # project_id, session_id (for UI to clear its view)
    activeSessionHistoryLoaded = Signal(str, str, list)  # project_id, session_id, history_list (for UI to load)
    messageChunkReceivedForSession = Signal(str, str, str, str)  # project_id, session_id, request_id, chunk_text
    messageFinalizedForSession = Signal(str, str, str, object, dict,
                                        bool)  # project_id, session_id, request_id, message_obj, usage_dict, is_error

    # Global UI Updates
    uiStatusUpdateGlobal = Signal(str, str, bool, int) # message, color_hex, is_temporary, duration_ms
    uiErrorGlobal = Signal(str, bool) # error_message, is_critical
    uiTextCopied = Signal(str, str) # message, color_hex
    uiInputBarBusyStateChanged = Signal(bool) # is_busy
    backendBusyStateChanged = Signal(bool) # is_busy
    ragStatusChanged = Signal(bool, str, str) # is_ready, status_text, status_color

    # Code Generation / Modification Flow
    modificationFileReadyForDisplay = Signal(str, str) # filename, content
    applyFileChangeRequested = Signal(str, str, str, str) # project_id, relative_filepath, new_content, focus_prefix

    @staticmethod
    def get_instance() -> 'EventBus':
        if EventBus._instance is None:
            EventBus._instance = EventBus()
        return EventBus._instance

    def __init__(self, parent: Optional[QObject] = None):
        if EventBus._instance is not None and id(self) != id(EventBus._instance):
            logger.warning(f"EventBus re-instantiated (ID: {id(self)}). Singleton ID: {id(EventBus._instance)}.")
        super().__init__(parent)
        if EventBus._instance is None:
            EventBus._instance = self
            logger.info(f"EventBus instance {id(self)} initialized (Parent: {parent}). This is the primary instance.")
        elif id(self) == id(EventBus._instance):
            pass # Already the primary instance

        logger.debug(f"EventBus instance {id(self)} signals defined.")

