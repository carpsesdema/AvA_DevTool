import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union, Dict, Any

try:
    from core.message_enums import MessageLoadingState
except ImportError:
    from enum import Enum, auto
    class MessageLoadingState(Enum): # type: ignore
        IDLE = auto()
        LOADING = auto()
        COMPLETED = auto()
        ERROR = auto()

USER_ROLE = "user"
MODEL_ROLE = "model"
SYSTEM_ROLE = "system"
ERROR_ROLE = "error"

@dataclass
class ChatMessage:
    role: str
    parts: List[Union[str, Dict[str, Any]]]
    timestamp: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    loading_state: MessageLoadingState = MessageLoadingState.IDLE

    @property
    def text(self) -> str:
        text_parts_list = []
        for part in self.parts:
            if isinstance(part, str):
                text_parts_list.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts_list.append(part.get("text", ""))
        return "".join(text_parts_list).strip()

    @property
    def has_images(self) -> bool:
        return any(isinstance(part, dict) and part.get("type") == "image" for part in self.parts)

    @property
    def image_parts(self) -> List[Dict[str, Any]]:
        return [part for part in self.parts if isinstance(part, dict) and part.get("type") == "image"]

