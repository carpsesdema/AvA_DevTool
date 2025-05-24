from enum import Enum, auto

class MessageLoadingState(Enum):
    IDLE = auto()
    LOADING = auto()
    COMPLETED = auto()
    ERROR = auto()

class ApplicationMode(Enum):
    IDLE = auto()
    NORMAL_CHAT_PROCESSING = auto()

