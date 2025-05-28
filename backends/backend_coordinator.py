import asyncio
import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple

from PySide6.QtCore import QObject, QTimer

try:
    from backends.backend_interface import BackendInterface
    from core.models import ChatMessage, MODEL_ROLE, USER_ROLE
    from core.event_bus import EventBus
except ImportError as e:
    logging.getLogger(__name__).error(f"Import error in backend_coordinator: {e}. Using placeholder types.")
    BackendInterface = type("BackendInterface", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})
    MODEL_ROLE, USER_ROLE = "model", "user"
    _dummy_signal = type("Signal", (object,), {"emit": lambda *args, **kwargs: None, "connect": lambda x: None})
    EventBus = type("EventBus", (object,), {
        "get_instance": lambda: type("DummyBus", (object,), {
            "llmRequestSent": _dummy_signal(),
            "llmStreamChunkReceived": _dummy_signal(),
            "llmResponseCompleted": _dummy_signal(),
            "llmResponseError": _dummy_signal(),
            "backendConfigurationChanged": _dummy_signal(),
            "backendBusyStateChanged": _dummy_signal(),
            "llmStreamStarted": _dummy_signal()
        })()
    })

logger = logging.getLogger(__name__)


class BackendCoordinator(QObject):
    def __init__(self, backend_adapters: Dict[str, BackendInterface], parent: Optional[QObject] = None):
        super().__init__(parent)
        if not backend_adapters:
            logger.critical("BackendCoordinator requires a non-empty dictionary of BackendInterface instances.")
            raise ValueError("BackendCoordinator requires at least one BackendInterface instance.")

        self._event_bus = EventBus.get_instance()
        self._backend_adapters = backend_adapters
        self._current_model_names: Dict[str, Optional[str]] = {bid: None for bid in backend_adapters}
        self._current_system_prompts: Dict[str, Optional[str]] = {bid: None for bid in backend_adapters}
        self._is_configured_map: Dict[str, bool] = {bid: False for bid in backend_adapters}
        self._available_models_map: Dict[str, List[str]] = {bid: [] for bid in backend_adapters}
        self._last_errors_map: Dict[str, Optional[str]] = {bid: None for bid in backend_adapters}
        self._active_backend_tasks: Dict[str, asyncio.Task] = {}
        self._overall_is_busy: bool = False

        self._models_fetch_cache: Dict[str, float] = {}
        self._models_fetch_cooldown = 30.0

        self._model_fetch_timer: Optional[QTimer] = None

        logger.info(
            f"BackendCoordinator initialized with {len(self._backend_adapters)} adapter(s): {list(self._backend_adapters.keys())}")

    def _update_overall_busy_state(self):
        new_busy_state = any(task and not task.done() for task in self._active_backend_tasks.values())
        if self._overall_is_busy != new_busy_state:
            self._overall_is_busy = new_busy_state
            self._event_bus.backendBusyStateChanged.emit(self._overall_is_busy)

    def configure_backend(self,
                          backend_id: str,
                          api_key: Optional[str],
                          model_name: str,
                          system_prompt: Optional[str] = None) -> bool:
        adapter = self._backend_adapters.get(backend_id)
        if not adapter:
            self._is_configured_map[backend_id] = False
            self._last_errors_map[backend_id] = f"Adapter not found for backend_id '{backend_id}'."
            self._current_model_names[backend_id] = model_name
            self._current_system_prompts[backend_id] = system_prompt
            self._available_models_map[backend_id] = []
            self._event_bus.backendConfigurationChanged.emit(backend_id, model_name, False, [])
            logger.error(self._last_errors_map[backend_id])
            return False

        is_configured = adapter.configure(api_key=api_key, model_name=model_name, system_prompt=system_prompt)
        self._is_configured_map[backend_id] = is_configured
        self._last_errors_map[backend_id] = adapter.get_last_error() if not is_configured else None
        self._current_model_names[backend_id] = model_name
        self._current_system_prompts[backend_id] = system_prompt

        cached_models = self._available_models_map.get(backend_id, [])
        self._event_bus.backendConfigurationChanged.emit(backend_id, model_name, is_configured, cached_models)

        if is_configured:
            self._schedule_async_model_fetch(backend_id)

        if not is_configured:
            logger.error(
                f"BC: Failed to configure backend '{backend_id}'. Last error: {self._last_errors_map[backend_id]}")

        return is_configured

    def _schedule_async_model_fetch(self, backend_id: str):
        """Schedule asynchronous model fetching to avoid blocking UI"""
        if not self._model_fetch_timer:
            self._model_fetch_timer = QTimer(self)
            self._model_fetch_timer.setSingleShot(True)
            self._model_fetch_timer.timeout.connect(lambda: self._fetch_models_async(backend_id))

        self._model_fetch_timer.start(100)

    def _fetch_models_async(self, backend_id: str):
        """Fetch models asynchronously for a specific backend"""
        import time
        current_time = time.time()

        if backend_id in self._models_fetch_cache:
            if current_time - self._models_fetch_cache[backend_id] < self._models_fetch_cooldown:
                logger.debug(f"BC: Skipping model fetch for '{backend_id}' due to cooldown")
                return

        adapter = self._backend_adapters.get(backend_id)
        if not adapter or not self._is_configured_map.get(backend_id, False):
            return

        try:
            available_models = adapter.get_available_models()
            self._available_models_map[backend_id] = available_models
            self._models_fetch_cache[backend_id] = current_time

            self._event_bus.backendConfigurationChanged.emit(
                backend_id,
                self._current_model_names[backend_id],
                True,
                available_models
            )
            logger.info(f"BC: Fetched {len(available_models)} models for '{backend_id}'")

        except Exception as e_fetch:
            logger.warning(f"BC: Failed to fetch models for '{backend_id}': {e_fetch}")
            cached_models = self._available_models_map.get(backend_id, [])
            self._event_bus.backendConfigurationChanged.emit(
                backend_id,
                self._current_model_names[backend_id],
                True,
                cached_models
            )

    def get_available_models_for_backend(self, backend_id: str) -> List[str]:
        """Get available models - return cached models immediately, fetch fresh in background"""
        cached_models = self._available_models_map.get(backend_id, [])

        # 🔥 DEBUG LOGGING
        logger.info(f"🐛 DEBUG: get_available_models_for_backend called for {backend_id}")
        logger.info(f"🐛 DEBUG: Cached models for {backend_id}: {cached_models}")

        adapter = self._backend_adapters.get(backend_id)
        if not adapter:
            logger.warning(f"🐛 DEBUG: No adapter found for {backend_id}")
            return cached_models

        import time
        current_time = time.time()
        should_refresh = (
                backend_id not in self._models_fetch_cache or
                current_time - self._models_fetch_cache[backend_id] > self._models_fetch_cooldown
        )

        logger.info(f"🐛 DEBUG: Should refresh models for {backend_id}? {should_refresh}")
        logger.info(f"🐛 DEBUG: Is configured: {self._is_configured_map.get(backend_id, False)}")

        if should_refresh and self._is_configured_map.get(backend_id, False):
            logger.info(f"🐛 DEBUG: Scheduling async model fetch for {backend_id}")
            self._schedule_async_model_fetch(backend_id)

        # 🔥 FORCE IMMEDIATE FETCH FOR DEBUGGING
        if backend_id.startswith("ollama") and self._is_configured_map.get(backend_id, False):
            logger.info(f"🐛 DEBUG: FORCE fetching models directly for {backend_id}")
            try:
                fresh_models = adapter.get_available_models()
                logger.info(f"🐛 DEBUG: Fresh models from adapter: {fresh_models}")
                self._available_models_map[backend_id] = fresh_models
                return fresh_models
            except Exception as e:
                logger.error(f"🐛 DEBUG: Error in force fetch: {e}")

        return cached_models

    def initiate_llm_chat_request(self,
                                  target_backend_id: str,
                                  history_to_send: List[ChatMessage],
                                  options: Optional[Dict[str, Any]] = None
                                  ) -> Tuple[bool, Optional[str], Optional[str]]:
        self._last_errors_map[target_backend_id] = None
        err_msg: Optional[str] = None
        adapter = self._backend_adapters.get(target_backend_id)
        if not adapter:
            err_msg = f"Adapter not found for backend_id '{target_backend_id}'."
            self._last_errors_map[target_backend_id] = err_msg
            return False, err_msg, None

        if not self._is_configured_map.get(target_backend_id, False):
            adapter_err = adapter.get_last_error()
            err_msg = f"Backend '{target_backend_id}' is not configured."
            if adapter_err:
                err_msg += f" Adapter msg: {adapter_err}"
            self._last_errors_map[target_backend_id] = err_msg
            return False, err_msg, None

        request_id = f"llm_req_{uuid.uuid4().hex[:12]}"
        logger.info(f"BC: Generated request ID: {request_id}")
        return True, None, request_id

    def start_llm_streaming_task(self,
                                 request_id: str,
                                 target_backend_id: str,
                                 history_to_send: List[ChatMessage],
                                 is_modification_response_expected: bool,
                                 options: Optional[Dict[str, Any]] = None,
                                 request_metadata: Optional[Dict[str, Any]] = None):
        adapter = self._backend_adapters.get(target_backend_id)
        if not adapter or not self._is_configured_map.get(target_backend_id, False):
            err = f"Cannot start stream for ReqID '{request_id}': Adapter '{target_backend_id}' not found/configured."
            self._event_bus.llmResponseError.emit(request_id, err)
            return

        if request_id in self._active_backend_tasks and not self._active_backend_tasks[request_id].done():
            return

        self._event_bus.llmRequestSent.emit(target_backend_id, request_id)
        try:
            task = asyncio.create_task(
                self._internal_get_response_stream(
                    backend_id=target_backend_id,
                    request_id=request_id,
                    adapter=adapter,
                    history=history_to_send,
                    options=options,
                    request_metadata=request_metadata
                )
            )
            self._active_backend_tasks[request_id] = task
            self._update_overall_busy_state()
        except Exception as e_create_task:
            logger.critical(
                f"BC: CRITICAL ERROR during asyncio.create_task for ReqID '{request_id}': {type(e_create_task).__name__} - {e_create_task}",
                exc_info=True)
            err_msg_detail = f"Failed to launch LLM task for '{target_backend_id}'. Error: {type(e_create_task).__name__}."
            self._last_errors_map[target_backend_id] = err_msg_detail
            self._event_bus.llmResponseError.emit(request_id, err_msg_detail)
            self._update_overall_busy_state()

    async def _internal_get_response_stream(self,
                                            backend_id: str,
                                            request_id: str,
                                            adapter: BackendInterface,
                                            history: List[ChatMessage],
                                            options: Optional[Dict[str, Any]] = None,
                                            request_metadata: Optional[Dict[str, Any]] = None):
        response_buffer = ""
        usage_stats_dict: Dict[str, Any] = {}
        if request_metadata:
            usage_stats_dict.update(request_metadata)
        usage_stats_dict["backend_id"] = backend_id
        usage_stats_dict["request_id"] = request_id

        project_id_for_event = "p1_chat_context"
        if request_metadata and "project_id" in request_metadata:
            project_id_for_event = request_metadata["project_id"]
        elif request_metadata and "p1_chat_context" in request_metadata:
            project_id_for_event = request_metadata["p1_chat_context"]
        usage_stats_dict["project_id"] = project_id_for_event

        try:
            if not hasattr(adapter, 'get_response_stream'):
                raise AttributeError(f"Adapter '{backend_id}' missing get_response_stream method.")

            self._event_bus.llmStreamStarted.emit(request_id)
            logger.info(f"BC: Started stream for request {request_id}")

            stream_iterator = adapter.get_response_stream(history, options)
            chunk_count = 0

            async for chunk in stream_iterator:
                chunk_count += 1
                logger.debug(f"BC: Emitting chunk #{chunk_count} for {request_id}: '{chunk[:30]}...'")
                self._event_bus.llmStreamChunkReceived.emit(request_id, chunk)
                response_buffer += chunk

                if chunk_count % 5 == 0:
                    await asyncio.sleep(0)

                if len(chunk) > 100:
                    await asyncio.sleep(0)

            logger.info(f"BC: Stream completed for {request_id}, total chunks: {chunk_count}")

            final_response_text = response_buffer.strip()
            token_usage = adapter.get_last_token_usage()
            if token_usage:
                usage_stats_dict["prompt_tokens"] = token_usage[0]
                usage_stats_dict["completion_tokens"] = token_usage[1]
            usage_stats_dict["model_name"] = getattr(adapter, "_model_name", "unknown")

            if final_response_text:
                completed_message = ChatMessage(id=request_id, role=MODEL_ROLE, parts=[final_response_text])
                logger.info(f"BC: Emitting completion for {request_id}")
                self._event_bus.llmResponseCompleted.emit(request_id, completed_message, usage_stats_dict)
            else:
                empty_msg_text = "[AI returned an empty response]"
                empty_message = ChatMessage(id=request_id, role=MODEL_ROLE, parts=[empty_msg_text])
                self._event_bus.llmResponseCompleted.emit(request_id, empty_message, usage_stats_dict)

        except asyncio.CancelledError:
            logger.info(f"BC: Stream cancelled for {request_id}")
            self._event_bus.llmResponseError.emit(request_id, "[AI response cancelled by user]")
        except Exception as e:
            error_msg = adapter.get_last_error() or f"Backend Task Error for ReqID {request_id}: {type(e).__name__}"
            self._last_errors_map[backend_id] = error_msg
            logger.error(f"BC: Stream error for {request_id}: {error_msg}")
            self._event_bus.llmResponseError.emit(request_id, error_msg)
        finally:
            self._active_backend_tasks.pop(request_id, None)
            self._update_overall_busy_state()

    def cancel_current_task(self, request_id: Optional[str] = None):
        if request_id:
            task = self._active_backend_tasks.get(request_id)
            if task and not task.done():
                task.cancel()
                self._update_overall_busy_state()
        else:
            for req_id_key, task_to_cancel in list(self._active_backend_tasks.items()):
                if task_to_cancel and not task_to_cancel.done():
                    task_to_cancel.cancel()
            self._update_overall_busy_state()

    def is_backend_configured(self, backend_id: str) -> bool:
        return self._is_configured_map.get(backend_id, False)

    def get_last_error_for_backend(self, backend_id: str) -> Optional[str]:
        adapter = self._backend_adapters.get(backend_id)
        direct_adapter_error = adapter.get_last_error() if adapter else None
        if direct_adapter_error:
            return direct_adapter_error
        return self._last_errors_map.get(backend_id)

    def is_any_backend_busy(self) -> bool:
        return self._overall_is_busy

    def get_current_configured_model(self, backend_id: str) -> Optional[str]:
        return self._current_model_names.get(backend_id)

    def get_current_system_prompt(self, backend_id: str) -> Optional[str]:
        return self._current_system_prompts.get(backend_id)

    def get_all_backend_ids(self) -> List[str]:
        """Returns a list of all backend IDs managed by the coordinator."""
        return list(self._backend_adapters.keys())