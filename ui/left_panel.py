import logging
from typing import Optional, Dict, List, Any

from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy,
    QComboBox, QGroupBox
)

try:
    import qtawesome as qta

    QTAWESOME_AVAILABLE = True
except ImportError:
    QTAWESOME_AVAILABLE = False
    qta = None
    logging.getLogger(__name__).warning("LeftControlPanel: qtawesome library not found. Icons will be limited.")

try:
    from utils import constants
    from core.event_bus import EventBus
    from core.chat_manager import ChatManager
except ImportError as e_lp:
    logging.getLogger(__name__).critical(f"Critical import error in LeftPanel: {e_lp}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class LeftControlPanel(QWidget):
    # This role stores a dict: {"backend_id": "...", "model_name": "..."}
    MODEL_CONFIG_DATA_ROLE = Qt.ItemDataRole.UserRole + 2

    def __init__(self, chat_manager: ChatManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LeftControlPanel")
        if not isinstance(chat_manager, ChatManager):
            logger.critical("LeftControlPanel requires a valid ChatManager instance.")
            raise TypeError("LeftControlPanel requires a valid ChatManager instance.")

        self.chat_manager = chat_manager
        self._event_bus = EventBus.get_instance()
        self._is_programmatic_model_change: bool = False

        self._init_widgets_phase1()
        self._init_layout_phase1()
        self._connect_signals_phase1()

        # Initial population and selection should happen after UI is fully set up
        # and after ChatManager has had a chance to initialize its backends.
        # This is now triggered by the backendConfigurationChanged event or in MainWindow's showEvent.
        # For now, we will call it directly after initialization to get initial state.
        self._load_initial_model_settings_phase1()
        logger.info("LeftControlPanel (Phase 1) initialized.")

    def _get_qta_icon(self, icon_name: str, color: str = "#00CFE8") -> QIcon:
        if QTAWESOME_AVAILABLE and qta:
            try:
                return qta.icon(icon_name, color=color)
            except Exception:
                pass
        return QIcon()

    def _init_widgets_phase1(self):
        self.button_font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1)
        button_style_sheet = "QPushButton { text-align: left; padding: 6px 8px; }"
        button_icon_size = QSize(16, 16)

        self.llm_config_group = QGroupBox("LLM Configuration")
        self.actions_group = QGroupBox("Actions")

        for group_box in [self.llm_config_group, self.actions_group]:
            group_box.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1, QFont.Weight.Bold))

        self.chat_llm_label = QLabel("Chat LLM:")
        self.chat_llm_label.setFont(self.button_font)

        self.chat_llm_combo_box = QComboBox()
        self.chat_llm_combo_box.setFont(self.button_font)
        self.chat_llm_combo_box.setObjectName("ChatLlmComboBox")
        self.chat_llm_combo_box.setToolTip("Select the primary AI model for chat")
        self.chat_llm_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.configure_ai_personality_button = QPushButton(" Configure Persona")
        self.configure_ai_personality_button.setFont(self.button_font)
        self.configure_ai_personality_button.setIcon(self._get_qta_icon('fa5s.user-cog', color="#DAA520"))
        self.configure_ai_personality_button.setToolTip("Customize AI personality / system prompt (Ctrl+P)")
        self.configure_ai_personality_button.setObjectName("configureAiPersonalityButton")
        self.configure_ai_personality_button.setStyleSheet(button_style_sheet)
        self.configure_ai_personality_button.setIconSize(button_icon_size)

        self.new_chat_button = QPushButton(" New Chat")
        self.new_chat_button.setFont(self.button_font)
        self.new_chat_button.setIcon(self._get_qta_icon('fa5s.comment-dots', color="#61AFEF"))
        self.new_chat_button.setToolTip("Start a new chat session (Ctrl+N)")
        self.new_chat_button.setObjectName("newChatButton")
        self.new_chat_button.setStyleSheet(button_style_sheet)
        self.new_chat_button.setIconSize(button_icon_size)

        self.view_llm_terminal_button = QPushButton(" View LLM Log")
        self.view_llm_terminal_button.setFont(self.button_font)
        self.view_llm_terminal_button.setIcon(self._get_qta_icon('fa5s.terminal', color="#98C379"))
        self.view_llm_terminal_button.setToolTip("Show LLM communication log (Ctrl+L)")
        self.view_llm_terminal_button.setObjectName("viewLlmTerminalButton")
        self.view_llm_terminal_button.setStyleSheet(button_style_sheet)
        self.view_llm_terminal_button.setIconSize(button_icon_size)

    def _init_layout_phase1(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        llm_config_layout = QVBoxLayout(self.llm_config_group)
        llm_config_layout.setSpacing(6)
        llm_config_layout.addWidget(self.chat_llm_label)
        llm_config_layout.addWidget(self.chat_llm_combo_box)
        llm_config_layout.addWidget(self.configure_ai_personality_button)
        main_layout.addWidget(self.llm_config_group)

        actions_layout = QVBoxLayout(self.actions_group)
        actions_layout.setSpacing(6)
        actions_layout.addWidget(self.new_chat_button)
        actions_layout.addWidget(self.view_llm_terminal_button)
        main_layout.addWidget(self.actions_group)

        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _connect_signals_phase1(self):
        self.new_chat_button.clicked.connect(lambda: self._event_bus.newChatRequested.emit())
        self.configure_ai_personality_button.clicked.connect(
            lambda: self._event_bus.chatLlmPersonalityEditRequested.emit())
        self.view_llm_terminal_button.clicked.connect(lambda: self._event_bus.showLlmLogWindowRequested.emit())
        self.chat_llm_combo_box.currentIndexChanged.connect(self._on_chat_llm_selected_phase1)

        # Connect to backendConfigurationChanged to update the model list
        self._event_bus.backendConfigurationChanged.connect(self._handle_backend_configuration_changed_event_phase1)
        # Connect to backendBusyStateChanged to update button states
        self._event_bus.backendBusyStateChanged.connect(self._handle_backend_busy_state_changed_event_phase1)

    def _load_initial_model_settings_phase1(self):
        # This method is called to initialize the combo box based on the current state.
        # It's defensive against multiple calls and ensures the UI reflects the backend.
        self._is_programmatic_model_change = True
        self.chat_llm_combo_box.blockSignals(True)

        # Populate with models from all available backends
        self._populate_chat_llm_combo_box_phase1()

        active_chat_backend_id = self.chat_manager.get_current_active_chat_backend_id()
        active_chat_model_name = self.chat_manager.get_model_for_backend(active_chat_backend_id)

        # Set the combo box to the currently active model
        self._set_combo_box_selection_phase1(self.chat_llm_combo_box, active_chat_backend_id, active_chat_model_name)

        self.chat_llm_combo_box.blockSignals(False)
        self._is_programmatic_model_change = False

        # Update tooltip and enabled states based on current ChatManager status
        self.update_personality_tooltip(active=bool(self.chat_manager.get_current_chat_personality()))
        self.set_enabled_state(enabled=self.chat_manager.is_api_ready(), is_busy=self.chat_manager.is_overall_busy())

    def _populate_chat_llm_combo_box_phase1(self):
        self.chat_llm_combo_box.clear()
        models_added_count = 0

        all_backend_ids = self.chat_manager.get_all_available_backend_ids()

        for backend_id in all_backend_ids:
            available_models_for_backend = self.chat_manager.get_available_models_for_backend(backend_id)

            # If no models are fetched, provide a default for that backend if it's the current active one
            if not available_models_for_backend:
                if backend_id == "gemini_chat_default":
                    available_models_for_backend = [constants.DEFAULT_GEMINI_CHAT_MODEL]
                elif backend_id == "ollama_chat_default":
                    available_models_for_backend = [constants.DEFAULT_OLLAMA_CHAT_MODEL]
                elif backend_id == "gpt_chat_default":
                    # Add a default for GPT if it's not configured
                    available_models_for_backend = ["gpt-4o", "gpt-3.5-turbo"]

            for model_name_str in available_models_for_backend:
                # Determine display name (e.g., "Gemini: 1.5-flash-latest", "Ollama: llama2:13b")
                display_name_prefix = ""
                if backend_id == "gemini_chat_default":
                    display_name_prefix = "Gemini: "
                    model_name_display = model_name_str.replace("models/", "")  # Remove "models/" prefix
                elif backend_id == "ollama_chat_default":
                    display_name_prefix = "Ollama: "
                    model_name_display = model_name_str  # Ollama model names are usually clean
                elif backend_id == "gpt_chat_default":
                    display_name_prefix = "GPT: "
                    model_name_display = model_name_str  # GPT model names are usually clean
                else:
                    model_name_display = model_name_str  # Fallback for unknown backends

                item_display_text = f"{display_name_prefix}{model_name_display}"
                user_data_for_item = {"backend_id": backend_id, "model_name": model_name_str}
                self.chat_llm_combo_box.addItem(item_display_text, userData=user_data_for_item)
                models_added_count += 1

        if models_added_count > 0:
            self.chat_llm_combo_box.setEnabled(True)
        else:
            self.chat_llm_combo_box.addItem("No Chat LLMs Available")
            self.chat_llm_combo_box.setEnabled(False)

    def _set_combo_box_selection_phase1(self, combo_box: QComboBox, target_backend_id: str,
                                        target_model_name: Optional[str]):
        # Iterate through items to find a match for both backend_id and model_name
        for i in range(combo_box.count()):
            item_data = combo_box.itemData(i)  # itemData() without role uses Qt.ItemDataRole.UserRole by default
            if isinstance(item_data, dict) and \
                    item_data.get("backend_id") == target_backend_id and \
                    item_data.get("model_name") == target_model_name:
                if combo_box.currentIndex() != i:
                    combo_box.setCurrentIndex(i)
                return

        # Fallback if exact model name not found (e.g., model no longer available, but backend is still active),
        # try to select any model for the specified backend.
        for i in range(combo_box.count()):
            item_data = combo_box.itemData(i)
            if isinstance(item_data, dict) and item_data.get("backend_id") == target_backend_id:
                if combo_box.currentIndex() != i:
                    combo_box.setCurrentIndex(i)
                return

        # If still nothing selected, try to set the default backend/model based on constants
        if combo_box.count() > 0:
            default_backend_id = constants.DEFAULT_CHAT_BACKEND_ID
            default_model_name = (constants.DEFAULT_OLLAMA_CHAT_MODEL if default_backend_id == "ollama_chat_default"
                                  else constants.DEFAULT_GEMINI_CHAT_MODEL)
            for i in range(combo_box.count()):
                item_data = combo_box.itemData(i)
                if isinstance(item_data, dict) and \
                        item_data.get("backend_id") == default_backend_id and \
                        item_data.get("model_name") == default_model_name:
                    combo_box.setCurrentIndex(i)
                    return
            combo_box.setCurrentIndex(0)  # Fallback to first item if all else fails

    @Slot(int)
    def _on_chat_llm_selected_phase1(self, index: int):
        if self._is_programmatic_model_change or index < 0:
            return

        selected_data = self.chat_llm_combo_box.itemData(index)  # Use default role
        if not isinstance(selected_data, dict) or \
                "backend_id" not in selected_data or \
                "model_name" not in selected_data:
            logger.warning(f"LP: Invalid item data selected in combo box: {selected_data}")
            return

        backend_id = selected_data["backend_id"]
        model_name = selected_data["model_name"]

        if self.chat_manager:
            logger.info(
                f"LP: User selected backend '{backend_id}', model '{model_name}'. Triggering chat manager config.")
            self.chat_manager.set_active_chat_model_and_backend(backend_id, model_name)

    @Slot(str, str, bool, list)
    def _handle_backend_configuration_changed_event_phase1(self, backend_id: str, model_name: str, is_configured: bool,
                                                           available_models: list):
        # This slot is triggered when any backend's configuration changes.
        # We need to re-populate the combo box to reflect potentially new available models
        # or changes in the configured status of the currently active model.
        logger.debug(f"LP: Backend config changed for '{backend_id}'. Updating combo box.")
        self._is_programmatic_model_change = True
        self.chat_llm_combo_box.blockSignals(True)

        # Get the currently active model from ChatManager before repopulating
        current_selected_backend = self.chat_manager.get_current_active_chat_backend_id()
        current_selected_model = self.chat_manager.get_model_for_backend(
            current_selected_backend)  # This gets the *configured* model

        self._populate_chat_llm_combo_box_phase1()

        # Re-set the selection to what was active (or default if it's no longer available)
        self._set_combo_box_selection_phase1(self.chat_llm_combo_box, current_selected_backend,
                                             current_selected_model)

        self.chat_llm_combo_box.blockSignals(False)
        self._is_programmatic_model_change = False
        self.update_personality_tooltip(active=bool(self.chat_manager.get_current_chat_personality()))

        # Update enabled state based on the overall chat manager status
        self.set_enabled_state(enabled=self.chat_manager.is_api_ready(), is_busy=self.chat_manager.is_overall_busy())

    @Slot(bool)
    def _handle_backend_busy_state_changed_event_phase1(self, is_busy: bool):
        # Update UI elements based on overall busy state of the backend coordinator
        self.set_enabled_state(enabled=self.chat_manager.is_api_ready(), is_busy=is_busy)

    def update_personality_tooltip(self, active: bool):
        tooltip_base = "Customize AI personality / system prompt (Ctrl+P)"
        status = "(Custom Persona Active)" if active else "(Default Persona)"
        self.configure_ai_personality_button.setToolTip(f"{tooltip_base}\nStatus: {status}")

    def set_enabled_state(self, enabled: bool, is_busy: bool):
        effective_enabled = enabled and not is_busy
        # LLM combo box should be enabled if *any* backend is configured
        self.chat_llm_combo_box.setEnabled(enabled)  # Keep enabled if at least one backend is ready
        self.configure_ai_personality_button.setEnabled(effective_enabled)
        self.new_chat_button.setEnabled(not is_busy)  # New chat always enabled unless busy
        self.view_llm_terminal_button.setEnabled(True)  # Log always viewable

        # Adjust label color based on overall API readiness
        label_color = "#C0C0C0" if enabled else "#707070"
        self.chat_llm_label.setStyleSheet(f"QLabel {{ color: {label_color}; }}")