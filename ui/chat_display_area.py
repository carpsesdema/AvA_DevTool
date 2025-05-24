# ui/chat_display_area.py
import logging
from typing import List, Optional, Tuple  # Added Tuple

from PySide6.QtCore import Qt, QTimer, Slot, QPoint, Signal, QModelIndex
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListView, QAbstractItemView, QMenu, QApplication
)

try:
    from core.models import ChatMessage, SYSTEM_ROLE, ERROR_ROLE
    from .chat_item_delegate import ChatItemDelegate  # Keep direct import
    from .chat_list_model import ChatListModel, ChatMessageRole  # Keep direct import
except ImportError as e_cda:
    logging.getLogger(__name__).critical(f"Critical import error in ChatDisplayArea: {e_cda}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatDisplayArea(QWidget):
    textCopied = Signal(str, str)  # message, color_hex

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatDisplayAreaWidget")
        self.chat_list_view: Optional[QListView] = None
        self.chat_list_model: Optional[ChatListModel] = None  # This model instance is specific to this display area
        self.chat_item_delegate: Optional[ChatItemDelegate] = None

        # Store the current project and session ID this display area is showing
        self._current_project_id: Optional[str] = None
        self._current_session_id: Optional[str] = None

        self._init_ui_phase1()  # Renamed
        # _connect_model_signals is not strictly needed if ChatListModel doesn't emit signals ChatDisplayArea needs directly.
        # Most updates will come via EventBus.

    def _init_ui_phase1(self):  # Renamed
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.chat_list_view = QListView(self)
        self.chat_list_view.setObjectName("ChatListView_Phase1")  # Renamed from ChatListView
        self.chat_list_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chat_list_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)  # Corrected enum
        self.chat_list_view.setResizeMode(QListView.ResizeMode.Adjust)  # Corrected enum
        self.chat_list_view.setUniformItemSizes(False)
        self.chat_list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)  # Corrected enum
        self.chat_list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.chat_list_model = ChatListModel(parent=self)  # Each display area gets its own model instance
        self.chat_list_view.setModel(self.chat_list_model)

        self.chat_item_delegate = ChatItemDelegate(parent=self)
        if self.chat_item_delegate:
            self.chat_item_delegate.setView(self.chat_list_view)
        self.chat_list_view.setItemDelegate(self.chat_item_delegate)

        self.chat_list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list_view.customContextMenuRequested.connect(self._show_chat_bubble_context_menu)  # Renamed

        outer_layout.addWidget(self.chat_list_view)
        self.setLayout(outer_layout)

    # MODIFIED: Methods now often take project_id and session_id to confirm context
    # However, a single ChatDisplayArea instance will usually only show one session at a time.
    # So, it mainly needs to check if the incoming event matches its *current* context.

    def set_current_context(self, project_id: str, session_id: str):  # ADDED
        logger.info(f"CDA: Setting current context to P:{project_id} S:{session_id}")
        self._current_project_id = project_id
        self._current_session_id = session_id
        # When context changes, typically the model is cleared and new history is loaded.
        # This is handled by _handle_active_session_history_loaded in MainWindow which calls loadHistory.

    @Slot(str, str, ChatMessage)  # MODIFIED: project_id, session_id, message
    def add_message_to_model(self, project_id: str, session_id: str, message: ChatMessage):
        if project_id == self._current_project_id and session_id == self._current_session_id:
            if self.chat_list_model:
                self.chat_list_model.addMessage(message)
                self._scroll_to_bottom_if_needed()  # MODIFIED
        else:
            logger.debug(
                f"CDA: Ignored add_message for P:{project_id}/S:{session_id} (current: P:{self._current_project_id}/S:{self._current_session_id})")

    @Slot(str,
          str)  # MODIFIED to take project/session context from ChatManager/EventBus if needed for future multi-session view
    def append_chunk_to_message_by_id(self, request_id: str, chunk_text: str):
        # This method is called by MainWindow's _handle_message_chunk_for_session
        # which already checks if the chunk is for the *active* session.
        if self.chat_list_model:
            success = self.chat_list_model.append_chunk_to_message_by_id(request_id, chunk_text)
            if success:
                self._scroll_to_bottom_if_needed(is_streaming=True)  # MODIFIED

    @Slot(str, ChatMessage, bool)  # MODIFIED: request_id, final_message_obj, is_error
    def finalize_message_by_id(self, request_id: str, final_message_obj: Optional[ChatMessage], is_error: bool):
        # Called by MainWindow's _handle_message_finalized_for_session for the active session.
        if self.chat_list_model:
            self.chat_list_model.finalize_message_by_id(request_id, final_message_obj, is_error)
            self._scroll_to_bottom_if_needed()  # MODIFIED

    @Slot(str, str)  # MODIFIED: project_id, session_id
    def clear_model_display(self, project_id: str, session_id: str):
        if project_id == self._current_project_id and session_id == self._current_session_id:
            if self.chat_list_model:
                logger.info(f"CDA: Clearing model for active P:{project_id}/S:{session_id}")
                self.chat_list_model.clearMessages()
                if self.chat_item_delegate: self.chat_item_delegate.clearCache()  # Clear delegate cache too
        else:
            logger.debug(
                f"CDA: Ignored clear_model for P:{project_id}/S:{session_id} (current: P:{self._current_project_id}/S:{self._current_session_id})")

    @Slot(str, str, list)  # MODIFIED: project_id, session_id, history
    def load_history_into_model(self, project_id: str, session_id: str, history: List[ChatMessage]):
        # This will be called by MainWindow when the active session's history is loaded.
        self.set_current_context(project_id, session_id)  # Ensure current context is set
        if self.chat_list_model:
            logger.info(f"CDA: Loading history ({len(history)} msgs) for P:{project_id}/S:{session_id}")
            self.chat_list_model.loadHistory(history)  # type: ignore
            if self.chat_item_delegate: self.chat_item_delegate.clearCache()
            self._scroll_to_bottom()  # Always scroll after loading full history

    def _scroll_to_bottom_if_needed(self, is_streaming: bool = False):  # MODIFIED
        if self.chat_list_view and self.chat_list_model and self.chat_list_model.rowCount() > 0:
            v_scrollbar = self.chat_list_view.verticalScrollBar()
            if v_scrollbar:
                # If streaming, only auto-scroll if user is already near the bottom.
                # If not streaming (e.g., new message added, history loaded), always scroll.
                should_scroll = True
                if is_streaming:
                    # Check if scrollbar is near the maximum value (e.g., within 1 page step or a fixed pixel threshold)
                    scroll_threshold = v_scrollbar.pageStep() // 2  # Heuristic
                    if v_scrollbar.value() < v_scrollbar.maximum() - scroll_threshold:
                        should_scroll = False

                if should_scroll:
                    QTimer.singleShot(0, self.chat_list_view.scrollToBottom)  # type: ignore

    def _scroll_to_bottom(self):  # ADDED (original simplified version)
        if self.chat_list_view and self.chat_list_model and self.chat_list_model.rowCount() > 0:
            QTimer.singleShot(0, self.chat_list_view.scrollToBottom)  # type: ignore

    def get_model(self) -> Optional[ChatListModel]:
        return self.chat_list_model

    def get_current_context(self) -> Tuple[Optional[str], Optional[str]]:  # ADDED
        return self._current_project_id, self._current_session_id

    def set_item_delegate(self, delegate: QAbstractItemView.itemDelegate):  # type: ignore # Corrected type
        if self.chat_list_view:
            self.chat_list_view.setItemDelegate(delegate)
            self.chat_item_delegate = delegate  # type: ignore
            if self.chat_item_delegate and hasattr(self.chat_item_delegate, 'setView'):
                self.chat_item_delegate.setView(self.chat_list_view)  # type: ignore

    @Slot(QPoint)  # Renamed
    def _show_chat_bubble_context_menu(self, pos: QPoint):
        if not self.chat_list_view or not self.chat_list_model: return

        index = self.chat_list_view.indexAt(pos)
        if not index.isValid(): return

        message = self.chat_list_model.data(index, ChatMessageRole)  # type: ignore
        if isinstance(message, ChatMessage) and \
                hasattr(message, 'role') and message.role not in [SYSTEM_ROLE, ERROR_ROLE] and \
                hasattr(message, 'text') and message.text and message.text.strip():
            context_menu = QMenu(self)
            copy_action = context_menu.addAction("Copy Message Text")
            copy_action.triggered.connect(
                lambda checked=False, msg_text=message.text: self._copy_message_text(msg_text))  # Renamed type: ignore
            context_menu.exec(self.chat_list_view.mapToGlobal(pos))

    def _copy_message_text(self, text: str):  # Renamed
        try:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text)
                self.textCopied.emit("Message text copied!", "#98c379")
        except Exception as e:
            logger.exception(f"Error copying text to clipboard: {e}")
            self.textCopied.emit(f"Error copying: {e}", "#e06c75")