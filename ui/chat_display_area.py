import logging
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, Slot, QPoint, Signal, QModelIndex
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListView, QAbstractItemView, QMenu, QApplication
)

try:
    from core.models import ChatMessage, SYSTEM_ROLE, ERROR_ROLE
    from .chat_item_delegate import ChatItemDelegate
    from .chat_list_model import ChatListModel, ChatMessageRole
except ImportError as e_cda:
    logging.getLogger(__name__).critical(f"Critical import error in ChatDisplayArea: {e_cda}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatDisplayArea(QWidget):
    textCopied = Signal(str, str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ChatDisplayAreaWidget")
        self.chat_list_view: Optional[QListView] = None
        self.chat_list_model: Optional[ChatListModel] = None
        self.chat_item_delegate: Optional[ChatItemDelegate] = None

        self._init_ui_phase1()
        self._connect_model_signals()

    def _init_ui_phase1(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.chat_list_view = QListView(self)
        self.chat_list_view.setObjectName("ChatListView_Phase1")
        self.chat_list_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chat_list_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.chat_list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.chat_list_view.setUniformItemSizes(False)
        self.chat_list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.chat_list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.chat_list_model = ChatListModel(parent=self)
        self.chat_list_view.setModel(self.chat_list_model)

        self.chat_item_delegate = ChatItemDelegate(parent=self)
        if self.chat_item_delegate:
            self.chat_item_delegate.setView(self.chat_list_view)
        self.chat_list_view.setItemDelegate(self.chat_item_delegate)

        self.chat_list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list_view.customContextMenuRequested.connect(self._show_chat_bubble_context_menu_phase1)

        outer_layout.addWidget(self.chat_list_view)
        self.setLayout(outer_layout)

    def _connect_model_signals(self):
        if self.chat_list_model:
            self.chat_list_model.modelReset.connect(self._handle_model_reset)
            self.chat_list_model.rowsInserted.connect(self._handle_rows_inserted)

    @Slot()
    def _handle_model_reset(self):
        if self.chat_item_delegate:
            self.chat_item_delegate.clearCache()
        self._scroll_to_bottom()

    @Slot(QModelIndex, int, int)
    def _handle_rows_inserted(self, parent: QModelIndex, first: int, last: int):
        self._scroll_to_bottom()

    @Slot(ChatMessage)  # type: ignore
    def add_message_to_model(self, message: ChatMessage):  # type: ignore
        if self.chat_list_model:
            self.chat_list_model.addMessage(message)

    @Slot(str, str)
    def append_chunk_to_message_by_id(self, message_id: str, chunk_text: str):
        if self.chat_list_model:
            success = self.chat_list_model.append_chunk_to_message_by_id(message_id, chunk_text)
            if success:
                v_scrollbar = self.chat_list_view.verticalScrollBar()  # type: ignore
                if v_scrollbar and v_scrollbar.value() >= v_scrollbar.maximum() - (v_scrollbar.pageStep() // 2):
                    self._scroll_to_bottom()

    @Slot(str, ChatMessage, bool)  # type: ignore
    def finalize_message_by_id(self, request_id: str, final_message_obj: Optional[ChatMessage] = None,
                               is_error: bool = False):  # type: ignore
        if self.chat_list_model:
            self.chat_list_model.finalize_message_by_id(request_id, final_message_obj, is_error)
            self._scroll_to_bottom()

    @Slot(list)
    def load_history_into_model(self, history: List[ChatMessage]):  # type: ignore
        if self.chat_list_model:
            self.chat_list_model.loadHistory(history)

    @Slot()
    def clear_model_display(self):
        if self.chat_list_model:
            self.chat_list_model.clearMessages()

    def _scroll_to_bottom(self):
        if self.chat_list_view and self.chat_list_model and self.chat_list_model.rowCount() > 0:
            QTimer.singleShot(0, self.chat_list_view.scrollToBottom)  # type: ignore

    def get_model(self) -> Optional[ChatListModel]:
        return self.chat_list_model

    def set_item_delegate(self, delegate: QAbstractItemView.itemDelegate):  # type: ignore
        if self.chat_list_view:
            self.chat_list_view.setItemDelegate(delegate)
            self.chat_item_delegate = delegate  # type: ignore
            if self.chat_item_delegate and hasattr(self.chat_item_delegate, 'setView'):
                self.chat_item_delegate.setView(self.chat_list_view)  # type: ignore

    @Slot(QPoint)
    def _show_chat_bubble_context_menu_phase1(self, pos: QPoint):
        if not self.chat_list_view or not self.chat_list_model: return

        index = self.chat_list_view.indexAt(pos)
        if not index.isValid(): return

        message = self.chat_list_model.data(index, ChatMessageRole)
        if isinstance(message, ChatMessage) and \
                hasattr(message, 'role') and message.role not in [SYSTEM_ROLE, ERROR_ROLE] and \
                hasattr(message, 'text') and message.text and message.text.strip():
            context_menu = QMenu(self)
            copy_action = context_menu.addAction("Copy Message Text")
            copy_action.triggered.connect(
                lambda checked=False, msg_text=message.text: self._copy_message_text_phase1(msg_text))  # type: ignore
            context_menu.exec(self.chat_list_view.mapToGlobal(pos))

    def _copy_message_text_phase1(self, text: str):
        try:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text)
                self.textCopied.emit("Message text copied!", "#98c379")
        except Exception as e:
            logger.exception(f"Error copying text to clipboard: {e}")
            self.textCopied.emit(f"Error copying: {e}", "#e06c75")
