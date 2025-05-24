from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

try:
    from core.models import ChatMessage
except ImportError:
    ChatMessage = "ChatMessage"

class BackendInterface(ABC):
    @abstractmethod
    def configure(self,
                  api_key: Optional[str],
                  model_name: str,
                  system_prompt: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    async def get_response_stream(self,
                                  history: List[ChatMessage], # type: ignore
                                  options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        if False: # type: ignore
            yield '' # type: ignore
        pass

    @abstractmethod
    def get_last_error(self) -> Optional[str]:
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass

    @abstractmethod
    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        pass

