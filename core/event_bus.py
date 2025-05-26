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
    # MODIFIED: Reverted to only dir_path for the global RAG scan
    requestRagScanDirectory = Signal(str) # for dir_path (targets GLOBAL RAG)

    # ADDED: Orchestrator-level request for new session
    createNewSessionForProjectRequested = Signal(str)  # project_id (ChatManager emits this)

    # UI Navigation / Dialog Triggers
    showLlmLogWindowRequested = Signal()
    chatLlmPersonalityEditRequested = Signal()
    viewCodeViewerRequested = Signal()
    createNewProjectRequested = Signal(str, str)
    openProjectSelectorRequested = Signal()
    renameCurrentSessionRequested = Signal(str)

    # Backend & LLM Communication
    backendConfigurationChanged = Signal(str, str, bool, list)
    llmRequestSent = Signal(str, str)
    llmStreamStarted = Signal(str)
    llmStreamChunkReceived = Signal(str, str)
    llmResponseCompleted = Signal(str, object, dict)
    llmResponseError = Signal(str, str)

    # Chat History & Session Management (MODIFIED to include project/session IDs)
    newMessageAddedToHistory = Signal(str, str, object)  # project_id, session_id, chat_message_obj
    activeSessionHistoryCleared = Signal(str, str)  # project_id, session_id (for UI to clear its view)
    activeSessionHistoryLoaded = Signal(str, str, list)  # project_id, session_id, history_list (for UI to load)
    messageChunkReceivedForSession = Signal(str, str, str, str)  # project_id, session_id, request_id, chunk_text
    messageFinalizedForSession = Signal(str, str, str, object, dict,
                                        bool)  # project_id, session_id, request_id, message_obj, usage_dict, is_error

    # Global UI Updates
    uiStatusUpdateGlobal = Signal(str, str, bool, int)
    uiErrorGlobal = Signal(str, bool)
    uiTextCopied = Signal(str, str)
    uiInputBarBusyStateChanged = Signal(bool)
    backendBusyStateChanged = Signal(bool)
    ragStatusChanged = Signal(bool, str, str) # RAG status still reflects project-specific readiness for display

    # Code Generation / Modification Flow
    modificationFileReadyForDisplay = Signal(str, str)
    applyFileChangeRequested = Signal(str, str, str, str)

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
            pass

        logger.debug(f"EventBus instance {id(self)} signals defined.")