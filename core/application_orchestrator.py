import logging
from typing import Dict, Optional, Any

from PySide6.QtCore import QObject

try:
    from backends.backend_interface import BackendInterface
    from backends.gemini_adapter import GeminiAdapter
    from backends.ollama_adapter import OllamaAdapter # ADDED: Import OllamaAdapter
    from backends.gpt_adapter import GPTAdapter # ADDED: Import GPTAdapter
    from backends.backend_coordinator import BackendCoordinator
    from core.event_bus import EventBus
    from services.llm_communication_logger import LlmCommunicationLogger
    from utils import constants
except ImportError as e:
    logging.getLogger(__name__).critical(f"Critical import error in ApplicationOrchestrator: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ApplicationOrchestrator(QObject):
    def __init__(self,
                 session_service_placeholder: Any,
                 upload_service_placeholder: Any,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ApplicationOrchestrator initializing (Phase 1)...")

        self.event_bus = EventBus.get_instance()
        if self.event_bus is None:
            logger.critical("EventBus instance is None in ApplicationOrchestrator.")
            raise RuntimeError("EventBus could not be instantiated.")

        self.gemini_chat_adapter = GeminiAdapter()
        self.ollama_chat_adapter = OllamaAdapter() # ADDED: Initialize OllamaAdapter
        self.gpt_chat_adapter = GPTAdapter() # ADDED: Initialize GPTAdapter


        self._all_backend_adapters_dict: Dict[str, BackendInterface] = {
            "gemini_chat_default": self.gemini_chat_adapter, # Explicitly list for Gemini
            "ollama_chat_default": self.ollama_chat_adapter, # ADDED: Add OllamaAdapter to dictionary
            "gpt_chat_default": self.gpt_chat_adapter # ADDED: Add GPTAdapter to dictionary
        }
        # The 'constants.DEFAULT_CHAT_BACKEND_ID' will resolve to one of the above keys.
        # This structure allows flexible default setting without redundancy.
        if constants.DEFAULT_CHAT_BACKEND_ID not in self._all_backend_adapters_dict:
            logger.warning(f"DEFAULT_CHAT_BACKEND_ID '{constants.DEFAULT_CHAT_BACKEND_ID}' not found in adapter map. Falling back to 'gemini_chat_default'.")
            # Fallback to ensure a default adapter is always present if constants.py is misconfigured
            self._all_backend_adapters_dict[constants.DEFAULT_CHAT_BACKEND_ID] = self.gemini_chat_adapter


        try:
            self.backend_coordinator = BackendCoordinator(self._all_backend_adapters_dict, parent=self)
        except ValueError as ve:
            logger.critical(f"Failed to instantiate BackendCoordinator: {ve}", exc_info=True)
            raise
        except Exception as e_bc:
            logger.critical(f"An unexpected error occurred instantiating BackendCoordinator: {e_bc}", exc_info=True)
            raise

        self.llm_communication_logger: Optional[LlmCommunicationLogger] = None
        try:
            self.llm_communication_logger = LlmCommunicationLogger(parent=self)
        except Exception as e_logger:
            logger.error(f"Failed to instantiate LlmCommunicationLogger: {e_logger}", exc_info=True)

        self._session_service_placeholder = session_service_placeholder
        self._upload_service_placeholder = upload_service_placeholder

        logger.info("ApplicationOrchestrator (Phase 1) initialization complete.")

    def get_event_bus(self) -> EventBus:
        return self.event_bus

    def get_backend_coordinator(self) -> BackendCoordinator:
        if not hasattr(self, 'backend_coordinator') or self.backend_coordinator is None:
            logger.critical("BackendCoordinator accessed before proper initialization in Orchestrator.")
            raise AttributeError("BackendCoordinator not initialized.")
        return self.backend_coordinator

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:
        return self.llm_communication_logger

    def get_all_backend_adapters_dict(self) -> Dict[str, BackendInterface]:
        return self._all_backend_adapters_dict

    def get_session_service_placeholder(self) -> Any:
        return self._session_service_placeholder

    def get_upload_service_placeholder(self) -> Any:
        return self._upload_service_placeholder