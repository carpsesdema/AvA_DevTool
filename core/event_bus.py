# core/event_bus.py
import logging
from typing import Optional
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class EventBus(QObject):
    _instance: Optional['EventBus'] = None

    userMessageSubmitted = Signal(str, list)
    newChatRequested = Signal()
    chatLlmPersonalitySubmitted = Signal(str, str)
    chatLlmSelectionChanged = Signal(str, str)
    specializedLlmSelectionChanged = Signal(str, str)

    showLlmLogWindowRequested = Signal()
    chatLlmPersonalityEditRequested = Signal()

    backendConfigurationChanged = Signal(str, str, bool, list)
    llmRequestSent = Signal(str, str)
    llmStreamStarted = Signal(str)
    llmStreamChunkReceived = Signal(str, str)
    llmResponseCompleted = Signal(str, object, dict)
    llmResponseError = Signal(str, str)

    newMessageAddedToHistory = Signal(str, object)
    activeSessionHistoryCleared = Signal(str)
    activeProjectChanged = Signal(str)

    uiStatusUpdateGlobal = Signal(str, str, bool, int)
    uiErrorGlobal = Signal(str, bool)
    uiTextCopied = Signal(str, str)
    uiInputBarBusyStateChanged = Signal(bool)
    backendBusyStateChanged = Signal(bool)

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