import logging
from typing import List, Optional, Any

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, QObject

try:
    from core.models import ChatMessage
    from core.message_enums import MessageLoadingState
except ImportError:
    # Fallback classes for type hinting if core imports fail,
    # allows the file to be parsed but will likely fail at runtime.
    class ChatMessage_fallback:
        pass
    ChatMessage = ChatMessage_fallback

    from enum import Enum, auto
    class MessageLoadingState_fallback(Enum):
        IDLE = auto()
        LOADING = auto()
        COMPLETED = auto()
        ERROR = auto()
    MessageLoadingState = MessageLoadingState_fallback
    logging.getLogger(__name__).warning(
        "ChatListModel: Could not import core.models or core.message_enums, using fallbacks.")

logger = logging.getLogger(__name__)

# Custom Roles for QAbstractListModel
ChatMessageRole = Qt.ItemDataRole.UserRole + 1 # Stores the actual ChatMessage object
LoadingStatusRole = Qt.ItemDataRole.UserRole + 2 # Stores the MessageLoadingState enum


class ChatListModel(QAbstractListModel):
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._messages: List[ChatMessage] = []
        logger.info("ChatListModel initialized.")

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        # Return 0 if the parent is valid (i.e., this is a child of an item, which we don't support)
        return 0 if parent.isValid() else len(self._messages)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._messages)):
            return None # Return None for invalid index

        message = self._messages[index.row()]

        if role == ChatMessageRole:
            return message # Return the full ChatMessage object
        elif role == Qt.ItemDataRole.DisplayRole:
            # For debugging/basic display, return a text preview
            text_preview = ""
            if hasattr(message, 'text') and message.text:
                text_preview = message.text[:50] + "..." if len(message.text) > 50 else message.text
            role_display = message.role if hasattr(message, 'role') else "unknown"
            return f"[{role_display}] {text_preview}"
        elif role == LoadingStatusRole:
            # Return the loading state for the delegate to render
            return message.loading_state if hasattr(message, 'loading_state') else MessageLoadingState.IDLE

        return None # For unhandled roles

    def addMessage(self, message: ChatMessage):
        if not isinstance(message, ChatMessage):  # type: ignore
            logger.error(f"Attempted to add invalid type to ChatListModel: {type(message)}")
            return

        row_to_insert = len(self._messages)
        self.beginInsertRows(QModelIndex(), row_to_insert, row_to_insert) # Notify view about impending rows
        self._messages.append(message)
        self.endInsertRows() # Notify view that rows have been inserted

    def append_chunk_to_message_by_id(self, message_id: str, chunk: str) -> bool:
        if not isinstance(message_id, str) or not message_id or not isinstance(chunk, str):
            return False

        row = self.find_message_row_by_id(message_id)
        if row is None:
            logger.warning(f"Model: Message with ID '{message_id}' not found to append chunk.")
            return False

        message = self._messages[row]

        # Ensure it's a model message (AI response) and has a 'parts' list
        if not hasattr(message, 'role') or message.role != "model":  # type: ignore
            logger.warning(f"Model: Attempted to append chunk to non-model message or message without role for ID '{message_id}'.")
            return False
        if not hasattr(message, 'parts') or not isinstance(message.parts, list):  # type: ignore
            message.parts = [] # Initialize if missing, though it should be a list for ChatMessage

        # Find or create the text part to append to
        current_text = ""
        text_part_index = -1
        for i, part_content in enumerate(message.parts):  # type: ignore
            if isinstance(part_content, str):
                current_text = part_content
                text_part_index = i
                break
            # If it's a dict and type is 'text', handle it. (Not standard for pure text ChatMessage parts,
            # but good for robustness if image parts are also present)
            elif isinstance(part_content, dict) and part_content.get("type") == "text":
                current_text = part_content.get("text", "")
                text_part_index = i
                break


        updated_text = current_text + chunk

        if text_part_index != -1:
            message.parts[text_part_index] = updated_text  # type: ignore
        else:
            # If no text part found, prepend a new one.
            message.parts.insert(0, updated_text)  # type: ignore

        # Set or update metadata indicating streaming
        if not hasattr(message, 'metadata') or message.metadata is None:  # type: ignore
            message.metadata = {}  # type: ignore
        message.metadata["is_streaming"] = True  # type: ignore

        # Notify the view that data for this item has changed
        model_idx = self.index(row, 0)
        self.dataChanged.emit(model_idx, model_idx, [ChatMessageRole, Qt.ItemDataRole.DisplayRole]) # Notify that ChatMessageRole and DisplayRole might have changed
        return True

    def finalize_message_by_id(self, message_id: str, final_message_obj: Optional[ChatMessage] = None,
                               is_error: bool = False):
        row = self.find_message_row_by_id(message_id)
        if row is None:
            logger.warning(f"Model: Message with ID '{message_id}' not found to finalize.")
            return

        message_to_update = self._messages[row]

        if final_message_obj and isinstance(final_message_obj, ChatMessage):  # type: ignore
            # If a final message object is provided, replace the existing one completely
            self._messages[row] = final_message_obj
            # Ensure loading state is set on the new object
            self._messages[row].loading_state = MessageLoadingState.ERROR if is_error else MessageLoadingState.COMPLETED  # type: ignore
        else:
            # If no final object, just update the existing message's state and metadata
            if hasattr(message_to_update, 'metadata') and message_to_update.metadata is not None:  # type: ignore
                message_to_update.metadata["is_streaming"] = False  # type: ignore
            if hasattr(message_to_update, 'loading_state'):
                message_to_update.loading_state = MessageLoadingState.ERROR if is_error else MessageLoadingState.COMPLETED  # type: ignore
            else:
                # If loading_state was missing, add it. This is a defensive measure.
                message_to_update.loading_state = MessageLoadingState.ERROR if is_error else MessageLoadingState.COMPLETED


        model_idx = self.index(row, 0)
        # Notify view about changes to ChatMessageRole (content update) and LoadingStatusRole (icon update)
        self.dataChanged.emit(model_idx, model_idx, [ChatMessageRole, LoadingStatusRole, Qt.ItemDataRole.DisplayRole])

    def updateMessage(self, index: int, message: ChatMessage):
        # This method is for general updates if an item at a specific index needs changing
        if not (0 <= index < len(self._messages)):
            logger.warning(f"Attempted to update message at invalid index: {index}")
            return
        if not isinstance(message, ChatMessage):  # type: ignore
            logger.error(f"Attempted to update message with invalid type: {type(message)}")
            return

        # Preserve the existing loading state if the new message is still IDLE
        existing_loading_state = MessageLoadingState.IDLE
        if hasattr(self._messages[index], 'loading_state'):
            existing_loading_state = self._messages[index].loading_state  # type: ignore

        self._messages[index] = message

        # If the new message is an AI response and was previously loading, keep its loading state
        if hasattr(self._messages[index], 'loading_state') and \
                hasattr(self._messages[index], 'role') and \
                self._messages[index].loading_state == MessageLoadingState.IDLE and \
                existing_loading_state != MessageLoadingState.IDLE and \
                self._messages[index].role == "model":  # type: ignore
            self._messages[index].loading_state = existing_loading_state  # type: ignore

        model_idx = self.index(index, 0)
        self.dataChanged.emit(model_idx, model_idx, [ChatMessageRole, LoadingStatusRole, Qt.ItemDataRole.DisplayRole])

    def update_message_loading_state_by_id(self, message_id: str, new_state: MessageLoadingState) -> bool:
        # This method is primarily used by ChatMessageStateHandler to update visual indicators
        if not isinstance(message_id, str) or not message_id:
            return False

        row = self.find_message_row_by_id(message_id)
        if row is not None:
            message = self._messages[row]
            # Only update if the state is actually changing
            if hasattr(message, 'loading_state') and message.loading_state != new_state:  # type: ignore
                message.loading_state = new_state  # type: ignore
                model_index = self.index(row, 0)
                self.dataChanged.emit(model_index, model_index, [LoadingStatusRole]) # Only notify LoadingStatusRole change
                return True
            # If loading_state wasn't set, or it's a model message, ensure it gets set.
            elif not hasattr(message, 'loading_state') and hasattr(message, 'role') and message.role == "model":  # type: ignore
                message.loading_state = new_state  # type: ignore
                model_index = self.index(row, 0)
                self.dataChanged.emit(model_index, model_index, [LoadingStatusRole])
                return True
        return False

    def find_message_row_by_id(self, message_id: str) -> Optional[int]:
        if not isinstance(message_id, str) or not message_id:
            return None
        for row_num, msg in enumerate(self._messages):
            if hasattr(msg, 'id') and msg.id == message_id:  # type: ignore
                return row_num
        return None

    def loadHistory(self, history: List[ChatMessage]):
        # Replaces the entire message list and notifies the view
        self.beginResetModel() # Notifies view that the entire model is about to be reset
        self._messages = list(history) if history else []
        self.endResetModel() # Notifies view that the model reset is complete

    def clearMessages(self):
        # Clears all messages and notifies the view
        self.beginResetModel()
        self._messages = []
        self.endResetModel()

    def getMessage(self, row: int) -> Optional[ChatMessage]:
        if 0 <= row < len(self._messages):
            return self._messages[row]
        return None

    def getAllMessages(self) -> List[ChatMessage]:
        return list(self._messages)