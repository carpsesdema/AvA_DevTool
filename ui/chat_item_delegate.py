import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import html  # Added for html.escape

from PySide6.QtCore import QModelIndex, QRect, QPoint, QSize, Qt, QObject, QByteArray, QPersistentModelIndex, Slot, \
    Signal
from PySide6.QtGui import (
    QPainter, QColor, QFontMetrics, QTextDocument, QPixmap, QImage, QFont,
    QMovie, QPen
)
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QStyleOptionViewItem, QWidget

try:
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from core.message_enums import MessageLoadingState
    from ui.chat_list_model import ChatMessageRole, LoadingStatusRole
    from utils import constants
except ImportError as e_delegate_import:
    logging.getLogger(__name__).critical(f"ChatItemDelegate: Critical import error: {e_delegate_import}", exc_info=True)
    raise

logger = logging.getLogger(__name__)

# Modern professional bubble dimensions
BUBBLE_PADDING_V = 14      # Generous but not excessive
BUBBLE_PADDING_H = 18      # Professional spacing
BUBBLE_MARGIN_V = 12       # Clean separation
BUBBLE_MARGIN_H = 16       # Professional margins
BUBBLE_RADIUS = 12         # Modern but not too playful
IMAGE_PADDING = 5
MAX_IMAGE_WIDTH = 250
MAX_IMAGE_HEIGHT = 250
MIN_BUBBLE_WIDTH = 60
USER_BUBBLE_INDENT_FACTOR = 0.20
TIMESTAMP_PADDING_TOP = 6  # Slightly more space for timestamps
TIMESTAMP_HEIGHT = 15
BUBBLE_MAX_WIDTH_PERCENTAGE = 0.75

INDICATOR_SIZE = QSize(18, 18)
INDICATOR_PADDING_X = 6
INDICATOR_PADDING_Y = 6

USER_BUBBLE_COLOR = QColor(
    constants.USER_BUBBLE_COLOR_HEX if hasattr(constants, 'USER_BUBBLE_COLOR_HEX') else "#0a7cff")
USER_TEXT_COLOR = QColor(constants.USER_TEXT_COLOR_HEX if hasattr(constants, 'USER_TEXT_COLOR_HEX') else "#ffffff")
AI_BUBBLE_COLOR = QColor(constants.AI_BUBBLE_COLOR_HEX if hasattr(constants, 'AI_BUBBLE_COLOR_HEX') else "#3E3E3E")
AI_TEXT_COLOR = QColor(constants.AI_TEXT_COLOR_HEX if hasattr(constants, 'AI_TEXT_COLOR_HEX') else "#E0E0E0")
SYSTEM_BUBBLE_COLOR = QColor(
    constants.SYSTEM_BUBBLE_COLOR_HEX if hasattr(constants, 'SYSTEM_BUBBLE_COLOR_HEX') else "#5A5A5A")
SYSTEM_TEXT_COLOR = QColor(
    constants.SYSTEM_TEXT_COLOR_HEX if hasattr(constants, 'SYSTEM_TEXT_COLOR_HEX') else "#B0B0B0")
ERROR_BUBBLE_COLOR = QColor(
    constants.ERROR_BUBBLE_COLOR_HEX if hasattr(constants, 'ERROR_BUBBLE_COLOR_HEX') else "#730202")
ERROR_TEXT_COLOR = QColor(constants.ERROR_TEXT_COLOR_HEX if hasattr(constants, 'ERROR_TEXT_COLOR_HEX') else "#FFCCCC")

BUBBLE_BORDER_COLOR = QColor(
    constants.BUBBLE_BORDER_COLOR_HEX if hasattr(constants, 'BUBBLE_BORDER_COLOR_HEX') else "#2D2D2D")
TIMESTAMP_COLOR = QColor(constants.TIMESTAMP_COLOR_HEX if hasattr(constants, 'TIMESTAMP_COLOR_HEX') else "#888888")
CODE_BLOCK_BG_COLOR = QColor(
    constants.CODE_BLOCK_BG_COLOR_HEX if hasattr(constants, 'CODE_BLOCK_BG_COLOR_HEX') else "#1E1E1E")


class ChatItemDelegate(QStyledItemDelegate):
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE)
        self._font_metrics = QFontMetrics(self._font)
        self._timestamp_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 2)
        self._timestamp_font_metrics = QFontMetrics(self._timestamp_font)

        self._text_doc_cache: Dict[Tuple[str, int, str], QTextDocument] = {}
        self._image_pixmap_cache: Dict[str, QPixmap] = {}

        self._loading_animation_movie_template: Optional[QMovie] = None
        self._completed_icon_pixmap: Optional[QPixmap] = None
        self._error_icon_pixmap: Optional[QPixmap] = None
        self._active_loading_movies: Dict[QPersistentModelIndex, QMovie] = {}
        self._view_ref: Optional[QWidget] = None

        self._init_indicator_assets()

    def _init_indicator_assets(self):
        try:
            loading_gif_path = os.path.join(constants.ASSETS_PATH, constants.LOADING_GIF_FILENAME)
            if os.path.exists(loading_gif_path):
                self._loading_animation_movie_template = QMovie(loading_gif_path)
                if self._loading_animation_movie_template.isValid():
                    self._loading_animation_movie_template.setScaledSize(INDICATOR_SIZE)
                else:
                    self._loading_animation_movie_template = None

            completed_png_path = os.path.join(constants.ASSETS_PATH, "loading_complete.png")
            if os.path.exists(completed_png_path):
                self._completed_icon_pixmap = QPixmap(completed_png_path).scaled(
                    INDICATOR_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )

            error_png_path = os.path.join(constants.ASSETS_PATH, "loading_error.png")
            if os.path.exists(error_png_path):
                self._error_icon_pixmap = QPixmap(error_png_path).scaled(
                    INDICATOR_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
        except Exception as e:
            logger.exception(f"Error initializing indicator assets: {e}")

    def setView(self, view: QWidget):
        self._view_ref = view

    @Slot(int)
    def _on_movie_frame_changed(self, frame_number: int):
        if not self._view_ref or not self._active_loading_movies:
            return

        movie_sender = self.sender()
        if not isinstance(movie_sender, QMovie):
            return

        for p_index, active_movie in list(self._active_loading_movies.items()):
            if active_movie == movie_sender:
                if p_index.isValid() and self._view_ref.model() and \
                        self._view_ref.model().data(p_index,
                                                    LoadingStatusRole) == MessageLoadingState.LOADING:  # type: ignore
                    self._view_ref.update(p_index)  # type: ignore
                else:
                    self._remove_active_movie(p_index)
                break

    def _remove_active_movie(self, p_index: QPersistentModelIndex):
        if p_index in self._active_loading_movies:
            movie = self._active_loading_movies.pop(p_index)
            movie.stop()
            try:
                movie.frameChanged.disconnect(self._on_movie_frame_changed)
            except TypeError:
                pass
            movie.deleteLater()

    def clearCache(self):
        self._text_doc_cache.clear()
        self._image_pixmap_cache.clear()
        for p_index in list(self._active_loading_movies.keys()):
            self._remove_active_movie(p_index)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        message = index.data(ChatMessageRole)
        if not isinstance(message, ChatMessage):
            super().paint(painter, option, index)
            painter.restore()
            return

        loading_status = index.model().data(index, LoadingStatusRole)  # type: ignore
        if not isinstance(loading_status, MessageLoadingState):
            loading_status = MessageLoadingState.IDLE

        is_user = (message.role == USER_ROLE)
        bubble_color, _ = self._get_colors(message.role)
        available_width = option.rect.width()

        content_size_info = self._calculate_content_size(message, available_width)
        bubble_rect = self._get_bubble_rect(option.rect, content_size_info["bubble_size"], is_user, available_width)

        painter.setPen(QPen(BUBBLE_BORDER_COLOR, 0.5))
        painter.setBrush(bubble_color)
        painter.drawRoundedRect(bubble_rect, BUBBLE_RADIUS, BUBBLE_RADIUS)

        content_draw_rect = bubble_rect.adjusted(BUBBLE_PADDING_H, BUBBLE_PADDING_V,
                                                 -BUBBLE_PADDING_H, -BUBBLE_PADDING_V)
        current_y_offset = 0

        if message.text:
            text_doc = self._get_prepared_text_document(message, content_draw_rect.width())
            painter.save()
            painter.translate(content_draw_rect.topLeft() + QPoint(0, current_y_offset))
            text_doc.drawContents(painter)
            painter.restore()
            current_y_offset += int(text_doc.size().height())

        if message.role == MODEL_ROLE:
            indicator_x = bubble_rect.right() - INDICATOR_SIZE.width() - INDICATOR_PADDING_X
            indicator_y = bubble_rect.top() + BUBBLE_PADDING_V
            indicator_rect = QRect(QPoint(indicator_x, indicator_y), INDICATOR_SIZE)
            persistent_index = QPersistentModelIndex(index)

            if loading_status == MessageLoadingState.LOADING:
                active_movie = self._active_loading_movies.get(persistent_index)
                if not active_movie and self._loading_animation_movie_template:
                    active_movie = QMovie(self._loading_animation_movie_template.fileName(), QByteArray(),
                                          self._view_ref or self)
                    if active_movie.isValid():
                        active_movie.setScaledSize(INDICATOR_SIZE)
                        active_movie.frameChanged.connect(self._on_movie_frame_changed)
                        self._active_loading_movies[persistent_index] = active_movie
                        active_movie.start()
                if active_movie and active_movie.isValid() and active_movie.state() == QMovie.MovieState.Running:
                    painter.drawPixmap(indicator_rect, active_movie.currentPixmap())

            elif loading_status == MessageLoadingState.COMPLETED:
                self._remove_active_movie(persistent_index)
                if self._completed_icon_pixmap:
                    painter.drawPixmap(indicator_rect, self._completed_icon_pixmap)

            elif loading_status == MessageLoadingState.ERROR:
                self._remove_active_movie(persistent_index)
                if self._error_icon_pixmap:
                    painter.drawPixmap(indicator_rect, self._error_icon_pixmap)
            else:
                self._remove_active_movie(persistent_index)

        formatted_timestamp = self._format_timestamp(message.timestamp)
        if formatted_timestamp:
            ts_y_pos = bubble_rect.bottom() + TIMESTAMP_PADDING_TOP + self._timestamp_font_metrics.ascent()
            if ts_y_pos < option.rect.bottom() - BUBBLE_MARGIN_V + 2:
                painter.setFont(self._timestamp_font)
                painter.setPen(TIMESTAMP_COLOR)
                ts_x_pos = bubble_rect.left() if message.role != USER_ROLE else \
                    bubble_rect.right() - self._timestamp_font_metrics.horizontalAdvance(formatted_timestamp)
                painter.drawText(QPoint(ts_x_pos, ts_y_pos), formatted_timestamp)

        if option.state & QStyle.StateFlag.State_Selected:
            highlight_color = option.palette.highlight().color()
            highlight_color.setAlpha(70)
            painter.fillRect(option.rect, highlight_color)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        message = index.data(ChatMessageRole)
        if not isinstance(message, ChatMessage):
            return super().sizeHint(option, index)

        available_width = option.rect.width()
        content_size_info = self._calculate_content_size(message, available_width)

        total_height = content_size_info["bubble_size"].height()

        if self._format_timestamp(message.timestamp):
            total_height += TIMESTAMP_PADDING_TOP + TIMESTAMP_HEIGHT

        final_height = total_height + 2 * BUBBLE_MARGIN_V

        min_text_line_height = self._font_metrics.height()
        min_bubble_content_height = min_text_line_height + 2 * BUBBLE_PADDING_V
        min_item_height = min_bubble_content_height + (
            TIMESTAMP_PADDING_TOP + TIMESTAMP_HEIGHT if self._format_timestamp(
                message.timestamp) else 0) + 2 * BUBBLE_MARGIN_V

        return QSize(available_width, max(final_height, min_item_height))

    def _get_colors(self, role: str) -> Tuple[QColor, QColor]:
        if role == USER_ROLE: return USER_BUBBLE_COLOR, USER_TEXT_COLOR
        if role == SYSTEM_ROLE: return SYSTEM_BUBBLE_COLOR, SYSTEM_TEXT_COLOR
        if role == ERROR_ROLE: return ERROR_BUBBLE_COLOR, ERROR_TEXT_COLOR
        return AI_BUBBLE_COLOR, AI_TEXT_COLOR

    def _get_bubble_rect(self, item_rect: QRect, bubble_content_qsize: QSize, is_user: bool,
                         available_item_width: int) -> QRect:
        bubble_w = bubble_content_qsize.width()
        bubble_h = bubble_content_qsize.height()

        if is_user:
            user_indent = int(available_item_width * USER_BUBBLE_INDENT_FACTOR)
            bubble_x = max(item_rect.left() + BUBBLE_MARGIN_H + user_indent,
                           item_rect.right() - BUBBLE_MARGIN_H - bubble_w)
        else:
            bubble_x = item_rect.left() + BUBBLE_MARGIN_H

        bubble_y = item_rect.top() + BUBBLE_MARGIN_V
        return QRect(bubble_x, bubble_y, bubble_w, bubble_h)

    def _calculate_content_size(self, message: ChatMessage, total_view_width: int) -> Dict[str, Any]:
        max_bubble_width_px = int(total_view_width * BUBBLE_MAX_WIDTH_PERCENTAGE)
        is_user = (message.role == USER_ROLE)
        if is_user:
            user_indent_px = int(total_view_width * USER_BUBBLE_INDENT_FACTOR)
            max_bubble_width_px = min(max_bubble_width_px, total_view_width - (2 * BUBBLE_MARGIN_H) - user_indent_px)

        max_bubble_width_px = max(max_bubble_width_px, MIN_BUBBLE_WIDTH + 2 * BUBBLE_PADDING_H)
        inner_content_width_constraint = max_bubble_width_px - (2 * BUBBLE_PADDING_H)

        current_content_height = 0
        max_content_width_used = 0

        if message.text:
            text_doc = self._get_prepared_text_document(message, inner_content_width_constraint)
            text_doc.setTextWidth(-1)
            ideal_text_size = text_doc.size()

            text_render_width = min(int(ideal_text_size.width()), inner_content_width_constraint)
            text_doc.setTextWidth(max(1, text_render_width))
            text_render_height = int(text_doc.size().height())

            current_content_height += text_render_height
            max_content_width_used = max(max_content_width_used, text_render_width)

        final_bubble_content_height = current_content_height + (2 * BUBBLE_PADDING_V)
        final_bubble_content_width = max(max_content_width_used, MIN_BUBBLE_WIDTH) + (2 * BUBBLE_PADDING_H)
        final_bubble_content_width = min(final_bubble_content_width, max_bubble_width_px)

        return {
            "bubble_size": QSize(final_bubble_content_width, final_bubble_content_height),
            "inner_content_width": max_content_width_used
        }

    def _get_prepared_text_document(self, message: ChatMessage, width_constraint: int) -> QTextDocument:
        text_content_for_cache = message.text if message.text else ""
        cache_key = (message.id, width_constraint, message.role)

        cached_doc = self._text_doc_cache.get(cache_key)
        if cached_doc:
            if abs(cached_doc.textWidth() - max(1, width_constraint)) > 1:
                cached_doc.setTextWidth(max(1, width_constraint))
            return cached_doc

        doc = QTextDocument()
        doc.setDefaultFont(self._font)
        doc.setDocumentMargin(0)

        _, text_color = self._get_colors(message.role)

        stylesheet_content = f"""
            body {{ color: {text_color.name()}; }}
            p {{ margin: 0 0 0px 0; padding: 0; line-height: 140%; }}
            ul, ol {{ margin: 2px 0 6px 18px; padding: 0; line-height: 140%;}}
            li {{ margin-bottom: 3px; }}
            pre {{ 
                background-color: {CODE_BLOCK_BG_COLOR.name()}; 
                border: 1px solid {BUBBLE_BORDER_COLOR.name()}; 
                padding: 8px; 
                margin: 6px 0; 
                border-radius: 4px; 
                font-family: '{self._font.family()}', monospace;
                font-size: {self._font.pointSize() - 1}pt;
                color: {AI_TEXT_COLOR.name()}; 
                white-space: pre-wrap; 
                word-wrap: break-word; 
                line-height: 130%;
            }}
            code {{ 
                background-color: {CODE_BLOCK_BG_COLOR.lighter(115).name()}; 
                padding: 1px 4px; 
                border-radius: 3px; 
                font-family: '{self._font.family()}', monospace;
                font-size: {int(self._font.pointSize() * 0.9)}pt;
                color: {AI_TEXT_COLOR.lighter(110).name()};
            }}
        """
        doc.setDefaultStyleSheet(stylesheet_content)

        html_content = self._convert_text_to_html(text_content_for_cache,
                                                  message.role == SYSTEM_ROLE or message.role == ERROR_ROLE)
        doc.setHtml(html_content)
        doc.setTextWidth(max(1, width_constraint))

        if len(self._text_doc_cache) > 150:
            self._text_doc_cache.pop(next(iter(self._text_doc_cache)))
        self._text_doc_cache[cache_key] = doc
        return doc

    def _convert_text_to_html(self, text: str, is_system_or_error: bool) -> str:
        if not text: return ""

        if is_system_or_error:
            escaped_text = html.escape(text).replace('\n', '<br/>')
            return f"<body><p>{escaped_text}</p></body>"

        try:
            import markdown
            md_extensions = ['fenced_code', 'nl2br', 'tables', 'sane_lists', 'extra']
            html_from_md = markdown.markdown(text, extensions=md_extensions)
            return f"<body>{html_from_md}</body>"
        except ImportError:
            escaped_text = html.escape(text).replace('\n', '<br/>')
            return f"<body><p>{escaped_text}</p></body>"
        except Exception as e_md:
            logger.error(f"Markdown conversion failed: {e_md}. Using basic escaping.")
            escaped_text = html.escape(text).replace('\n', '<br/>')
            return f"<body><p>{escaped_text}</p></body>"

    def _format_timestamp(self, iso_timestamp: Optional[str]) -> Optional[str]:
        if not iso_timestamp: return None
        try:
            dt_object = datetime.fromisoformat(iso_timestamp)
            return dt_object.strftime("%H:%M")
        except (ValueError, TypeError):
            return None