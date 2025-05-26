# services/llm_communication_logger.py
import html
import logging
import re
from datetime import datetime
from typing import Optional, Set, Dict, Any

from PySide6.QtCore import QObject, Signal, Slot

# --- Pygments Imports for Syntax Highlighting ---
try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer, PythonLexer
    from pygments.formatters import HtmlFormatter
    from pygments.styles import get_style_by_name
    from pygments.util import ClassNotFound as PygmentsClassNotFound

    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    highlight = None;
    get_lexer_by_name = None;
    guess_lexer = None;
    PythonLexer = None  # type: ignore
    HtmlFormatter = None;
    get_style_by_name = None;
    PygmentsClassNotFound = type("PygmentsClassNotFound", (Exception,), {})  # type: ignore
    logging.getLogger(__name__).warning(
        "LlmCommunicationLogger: 'Pygments' library not found. Code block syntax highlighting will be basic."
    )

try:
    from core.event_bus import EventBus
except ImportError:
    EventBus = None
    logging.getLogger(__name__).warning("LlmCommunicationLogger: EventBus not available")

logger = logging.getLogger(__name__)

STYLED_CONTENT_SENDERS = {"PLANNER AI", "CODE LLM", "GEMINI", "OLLAMA", "GPT", "USER", "TERMINAL"}
META_PREFIX_KEYWORDS: Set[str] = {
    "[MC SYSTEM]", "[SYSTEM PROCESS]", "[SYSTEM INFO]", "[SYSTEM]",
    "[RAG QUERY]", "[PROCESS]", "[INFO]", "[DEBUG]",
    "[ERROR", "[WARN",
    "EXTRACTED CODER INSTRUCTIONS FOR:", "PLANNED FILES:", "RECEIVED PLAN. LENGTH:",
    "PARSING...", "SENDING REQUEST TO", "GENERATING CODE FOR"
}

HTML_SENDER_STYLES: Dict[str, Dict[str, Any]] = {
    "ERROR": {"prefix_color": "#FF6B6B", "symbol": "❌ ", "is_bold": True},
    "WARN": {"prefix_color": "#FFA94D", "symbol": "⚠️ ", "is_bold": True},
    "PLANNER AI": {"prefix_color": "#4DFFFF", "content_color": "#B0E0E6", "symbol": "🧠 ", "is_bold": False},
    "GEMINI": {"prefix_color": "#87CEFA", "content_color": "#ADD8E6", "symbol": "✨ ", "is_bold": False},
    "CODE LLM": {"prefix_color": "#73F073", "content_color": "#90EE90", "symbol": "💻 ", "is_bold": False},
    "OLLAMA": {"prefix_color": "#FFB6C1", "content_color": "#FFC0CB", "symbol": "🦙 ", "is_bold": False},
    "GPT": {"prefix_color": "#9370DB", "content_color": "#B19CD9", "symbol": "💡 ", "is_bold": False},
    "SYSTEM": {"prefix_color": "#FFFACD", "symbol": "⚙️ ", "is_bold": True},
    "USER": {"prefix_color": "#98FB98", "content_color": "#98FB98", "symbol": "> ", "is_bold": True},
    "RAG": {"prefix_color": "#FFC0CB", "symbol": "📚 ", "is_bold": True},
    "PROCESS": {"prefix_color": "#ADD8E6", "symbol": "🛠️ ", "is_bold": True},
    "INFO": {"prefix_color": "#D3D3D3", "symbol": "ℹ️ ", "is_bold": False},
    "DEBUG": {"prefix_color": "#A9A9A9", "symbol": "🐞 ", "is_bold": False},
    # NEW: Terminal-related styles
    "TERMINAL": {"prefix_color": "#00FF41", "content_color": "#00FF41", "symbol": "$ ", "is_bold": True},
    "TERMINAL_OUT": {"prefix_color": "#CCCCCC", "content_color": "#CCCCCC", "symbol": "  ", "is_bold": False},
    "TERMINAL_ERR": {"prefix_color": "#FF4444", "content_color": "#FF4444", "symbol": "⚠ ", "is_bold": False},
    "TERMINAL_OK": {"prefix_color": "#44FF44", "content_color": "#44FF44", "symbol": "✓ ", "is_bold": True},
    "TERMINAL_FAIL": {"prefix_color": "#FF4444", "content_color": "#FF4444", "symbol": "✗ ", "is_bold": True},
}

HTML_DEFAULT_SENDER_STYLE: Dict[str, Any] = {"prefix_color": "#E0E0E0", "symbol": "💬 ", "is_bold": False}
HTML_TIMESTAMP_COLOR = "#686868"
HTML_DEFAULT_MESSAGE_COLOR = "#DCDCDC"
HTML_META_MESSAGE_COLOR = "#888C8F"
HTML_META_FONT_SIZE = "0.9em"
PYGMENTS_STYLE_NAME = 'material'
CODE_BLOCK_REGEX = re.compile(r"```(?:([a-zA-Z0-9_\-.+#\s]*)\n)?(.*?)```", re.DOTALL)


class LlmCommunicationLogger(QObject):
    new_terminal_log_entry = Signal(str)
    _html_formatter: Optional[HtmlFormatter] = None
    _pygments_style_defines_bg = False

    if PYGMENTS_AVAILABLE and HtmlFormatter:
        try:
            _pyg_style_obj = get_style_by_name(PYGMENTS_STYLE_NAME)
            if _pyg_style_obj.background_color is not None:
                _pygments_style_defines_bg = True
            _html_formatter = HtmlFormatter(
                style=PYGMENTS_STYLE_NAME,
                noclasses=True,
                wrapcode=False,
                lineseparator="<br/>"
            )
            logger.info(
                f"Pygments HtmlFormatter initialized with style '{PYGMENTS_STYLE_NAME}'. "
                f"Style defines background: {_pygments_style_defines_bg}"
            )
        except Exception as e_pyg_init:
            logger.error(f"Failed to initialize Pygments HtmlFormatter: {e_pyg_init}", exc_info=True)
            _html_formatter = None
            PYGMENTS_AVAILABLE = False

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._event_bus = EventBus.get_instance() if EventBus else None
        self._connect_terminal_signals()
        logger.info("LlmCommunicationLogger initialized.")

    def _connect_terminal_signals(self):
        """Connect to terminal-related EventBus signals"""
        if not self._event_bus:
            return

        self._event_bus.terminalCommandStarted.connect(self._handle_terminal_command_started)
        self._event_bus.terminalCommandOutput.connect(self._handle_terminal_output)
        self._event_bus.terminalCommandCompleted.connect(self._handle_terminal_completed)
        self._event_bus.terminalCommandError.connect(self._handle_terminal_error)

    @Slot(str, str)
    def _handle_terminal_command_started(self, command_id: str, command: str):
        """Handle terminal command start"""
        self.log_message("TERMINAL", f"{command}")

    @Slot(str, str, str)
    def _handle_terminal_output(self, command_id: str, output_type: str, content: str):
        """Handle terminal command output (stdout/stderr)"""
        if not content.strip():
            return

        if output_type == "stdout":
            self.log_message("TERMINAL_OUT", content)
        elif output_type == "stderr":
            self.log_message("TERMINAL_ERR", content)

    @Slot(str, int, float)
    def _handle_terminal_completed(self, command_id: str, exit_code: int, execution_time: float):
        """Handle terminal command completion"""
        if exit_code == 0:
            self.log_message("TERMINAL_OK", f"Command completed successfully ({execution_time:.2f}s)")
        else:
            self.log_message("TERMINAL_FAIL", f"Command failed with exit code {exit_code} ({execution_time:.2f}s)")

    @Slot(str, str)
    def _handle_terminal_error(self, command_id: str, error_message: str):
        """Handle terminal command error"""
        self.log_message("TERMINAL_FAIL", f"Command error: {error_message}")

    def _is_meta_message(self, sender_prefix_str: str, message_str: str) -> bool:
        prefix_upper = sender_prefix_str.upper()
        message_upper = message_str.upper()
        for keyword in META_PREFIX_KEYWORDS:
            if prefix_upper.startswith(keyword.upper()):
                return True
        if "PLANNER AI" in prefix_upper or "GEMINI" in prefix_upper:
            if any(status_phrase in message_upper for status_phrase in
                   ["EXTRACTED CODER INSTRUCTIONS FOR:", "PLANNED FILES:", "RECEIVED PLAN. LENGTH:", "PARSING...",
                    "SENDING REQUEST TO", "GENERATING CODE FOR"]):
                return True
        return False

    def _highlight_code_block_html(self, code_content: str, language_hint: str = "") -> str:
        if not PYGMENTS_AVAILABLE or not self._html_formatter or not highlight:
            escaped_code = html.escape(code_content)
            return (f'<pre style="background-color: #1E1E1E; color: #DCDCDC; padding: 10px; margin: 8px 0; '
                    f'border-radius: 4px; border: 1px solid #333; white-space: pre-wrap; word-wrap: break-word; '
                    f'font-family: monospace; line-height: 1.4;">{escaped_code}</pre>')
        try:
            lexer = None
            if language_hint:
                try:
                    hint_lower = language_hint.strip().lower()
                    if hint_lower in ["python", "py"]:
                        lexer = PythonLexer()
                    elif hint_lower:
                        lexer = get_lexer_by_name(hint_lower)
                except PygmentsClassNotFound:
                    logger.debug(f"Pygments lexer not found for hint: '{language_hint}'. Will try guessing.")
            if not lexer:
                try:
                    lexer = guess_lexer(code_content)
                except PygmentsClassNotFound:
                    logger.debug("Pygments could not guess lexer. Defaulting to PythonLexer for code block.")
                    lexer = PythonLexer()
            highlighted_code_html = highlight(code_content, lexer, self._html_formatter)
            pre_match = re.search(r"<pre.*?>(.*?)</pre>", highlighted_code_html, re.DOTALL | re.IGNORECASE)
            inner_highlighted_code = pre_match.group(1) if pre_match else highlighted_code_html
            pre_bg_color = "transparent" if self._pygments_style_defines_bg else "#1E1E1E"
            pre_border = "none" if self._pygments_style_defines_bg else "1px solid #383838"
            return (f'<pre style="background-color: {pre_bg_color}; color: #DCDCDC; padding: 10px; margin: 8px 0; '
                    f'border-radius: 4px; border: {pre_border}; white-space: pre-wrap; word-wrap: break-word; '
                    f'font-family: monospace; line-height: 1.45;">{inner_highlighted_code}</pre>')
        except Exception as e_highlight:
            logger.error(f"Error during Pygments syntax highlighting: {e_highlight}", exc_info=True)
            escaped_code = html.escape(code_content)
            return (f'<pre style="background-color: #1E1E1E; color: #DCDCDC; padding: 10px; margin: 8px 0; '
                    f'border-radius: 4px; border: 1px solid #333; white-space: pre-wrap; word-wrap: break-word; '
                    f'font-family: monospace; line-height: 1.4;">{escaped_code}</pre>')

    def log_message(self, sender_prefix: str, message: str):
        if not isinstance(message, str) or not message.strip():
            if sender_prefix and sender_prefix.strip():
                logger.debug(f"LLMCommLogger: Logging message from '{sender_prefix}' with empty content.")
            else:
                return
        timestamp_dt = datetime.now()
        html_timestamp_str = timestamp_dt.strftime("%H:%M:%S.%f")[:-3]
        is_meta = self._is_meta_message(sender_prefix, message)
        chosen_style = HTML_DEFAULT_SENDER_STYLE
        prefix_upper_no_brackets = sender_prefix.upper().replace('[', '').replace(']', '')
        for keyword, style_dict_val in HTML_SENDER_STYLES.items():
            if keyword in prefix_upper_no_brackets:
                chosen_style = style_dict_val
                break
        sender_symbol = chosen_style.get("symbol", "• ")
        prefix_color = chosen_style.get("prefix_color", HTML_DEFAULT_SENDER_STYLE["prefix_color"])
        is_prefix_bold = chosen_style.get("is_bold", False)
        if is_meta and not (sender_prefix.upper().startswith("[ERROR") or sender_prefix.upper().startswith("[WARN")):
            prefix_color = HTML_META_MESSAGE_COLOR
        message_content_color = HTML_DEFAULT_MESSAGE_COLOR
        if is_meta:
            message_content_color = HTML_META_MESSAGE_COLOR
        elif sender_prefix.upper().replace('[', '').replace(']', '') in STYLED_CONTENT_SENDERS:
            message_content_color = chosen_style.get("content_color", HTML_DEFAULT_MESSAGE_COLOR)
        formatted_message_parts = []
        last_end_pos = 0
        message_to_process = message.strip()
        for match in CODE_BLOCK_REGEX.finditer(message_to_process):
            pre_code_text = message_to_process[last_end_pos:match.start()]
            if pre_code_text.strip():
                formatted_message_parts.append(html.escape(pre_code_text).replace('\n', '<br/>'))
            lang_hint = match.group(1).strip() if match.group(1) else ""
            code_content = match.group(2)
            formatted_message_parts.append(self._highlight_code_block_html(code_content, lang_hint))
            last_end_pos = match.end()
        remaining_text = message_to_process[last_end_pos:]
        if remaining_text.strip():
            formatted_message_parts.append(html.escape(remaining_text).replace('\n', '<br/>'))
        final_formatted_message_content = "".join(formatted_message_parts)
        if not final_formatted_message_content.strip() and message.strip():
            final_formatted_message_content = html.escape(message.strip()).replace('\n', '<br/>')
        div_style_parts = [
            "padding: 1px 5px;",
            "margin-bottom: 1px;",
            "line-height: 1.45;"
        ]
        if is_meta:
            div_style_parts.append(f"font-size: {HTML_META_FONT_SIZE};")
        html_entry = (
            f'<div style="{" ".join(div_style_parts)}">'
            f'<span style="color: {HTML_TIMESTAMP_COLOR}; font-size: 0.85em; margin-right: 8px;">[{html_timestamp_str}]</span>'
            f'<span style="color: {prefix_color}; font-weight: {"bold" if is_prefix_bold else "normal"};">{sender_symbol}{html.escape(sender_prefix)}:</span> '
            f'<span style="color: {message_content_color};">{final_formatted_message_content}</span>'
            f'</div>'
        )
        self.new_terminal_log_entry.emit(html_entry)