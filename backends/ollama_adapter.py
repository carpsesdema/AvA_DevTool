import asyncio
import base64
import logging
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple, Type

try:
    import ollama

    _ollama_module_present = True
    try:
        from ollama._types import Model as _OllamaModelType_Imported  # type: ignore
        from ollama._types import ListResponse as _OllamaListResponseType_Imported  # type: ignore

        OllamaModelType: Optional[Type] = _OllamaModelType_Imported
        OllamaListResponseType: Optional[Type] = _OllamaListResponseType_Imported
        _ollama_types_imported_successfully = True
    except ImportError:
        OllamaModelType = None
        OllamaListResponseType = None
        _ollama_types_imported_successfully = False
    API_LIBRARY_AVAILABLE = True
except ImportError:
    ollama = None  # type: ignore
    _ollama_module_present = False
    OllamaModelType = None
    OllamaListResponseType = None
    _ollama_types_imported_successfully = False
    API_LIBRARY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "OllamaAdapter: 'ollama' library not found. Please install it: pip install ollama")

try:
    from backends.backend_interface import BackendInterface
    from core.models import ChatMessage, MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE
except ImportError:
    BackendInterface = type("BackendInterface", (object,), {})  # type: ignore
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE = "model", "user", "system", "error"

logger = logging.getLogger(__name__)


class OllamaAdapter(BackendInterface):
    DEFAULT_OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(self):
        super().__init__()
        self._sync_client: Optional[ollama.Client] = None
        self._model_name: str = self.DEFAULT_MODEL
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._ollama_host: str = self.DEFAULT_OLLAMA_HOST
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        logger.info("OllamaAdapter initialized.")

    def configure(self, api_key: Optional[str], model_name: Optional[str], system_prompt: Optional[str] = None) -> bool:
        self._sync_client = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library ('ollama') not installed."
            return False

        self._model_name = model_name if model_name and model_name.strip() else self.DEFAULT_MODEL
        self._system_prompt = system_prompt.strip() if isinstance(system_prompt,
                                                                  str) and system_prompt.strip() else None

        try:
            self._sync_client = ollama.Client(host=self._ollama_host)  # type: ignore
            try:
                self._sync_client.list()  # type: ignore
            except ollama.RequestError as conn_err_req:  # type: ignore
                self._last_error = f"Failed to connect to Ollama at {self._ollama_host} (RequestError): {conn_err_req}"
                self._sync_client = None
                return False
            except ConnectionRefusedError:
                self._last_error = f"Connection refused by Ollama at {self._ollama_host}. Is Ollama running?"
                self._sync_client = None
                return False
            except Exception as conn_err_other:
                self._last_error = f"Failed to connect/verify Ollama at {self._ollama_host}: {type(conn_err_other).__name__} - {conn_err_other}"
                self._sync_client = None
                return False
            self._is_configured = True
            return True
        except Exception as e:
            self._last_error = f"Unexpected error configuring Ollama client: {type(e).__name__} - {e}"
            self._sync_client = None
            return False

    def is_configured(self) -> bool:
        return self._is_configured and self._sync_client is not None

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> \
            AsyncGenerator[str, None]:  # type: ignore
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._sync_client:
            self._last_error = "Adapter is not configured."
            raise RuntimeError(self._last_error)

        messages_for_api = self._format_history_for_api(history)  # type: ignore
        if not messages_for_api and not self._system_prompt:  # Allow empty if system prompt is set
            self._last_error = "Cannot send request: No valid messages in history for API format and no system prompt."
            raise ValueError(self._last_error)

        ollama_api_options: Dict[str, Any] = {}
        if options:
            if "temperature" in options and isinstance(options["temperature"], (float, int)):
                ollama_api_options["temperature"] = float(options["temperature"])

        try:
            def _blocking_ollama_stream_call():
                return self._sync_client.chat(  # type: ignore
                    model=self._model_name,
                    messages=messages_for_api,
                    stream=True,
                    options=ollama_api_options
                )

            # Get the stream iterator
            stream_iterator = await asyncio.to_thread(_blocking_ollama_stream_call)

            # FIXED: Process stream chunk by chunk instead of all at once
            chunk_count = 0
            for chunk in stream_iterator:
                chunk_count += 1

                if not isinstance(chunk, dict):
                    continue

                if chunk.get("error"):
                    error_msg = chunk["error"]
                    self._last_error = error_msg
                    yield f"[SYSTEM ERROR: {error_msg}]"
                    if chunk.get('done', False):
                        self._last_prompt_tokens = chunk.get('prompt_eval_count')
                        self._last_completion_tokens = chunk.get('eval_count')
                    return

                content_part = chunk.get('message', {}).get('content', '')
                if content_part:
                    yield content_part

                # CRITICAL: Yield control every few chunks to prevent GUI blocking
                if chunk_count % 3 == 0:
                    await asyncio.sleep(0)

                if chunk.get('done', False):
                    self._last_prompt_tokens = chunk.get('prompt_eval_count')
                    self._last_completion_tokens = chunk.get('eval_count')
                    break

        except ollama.ResponseError as e_resp:  # type: ignore
            self._last_error = f"Ollama API Response Error: {e_resp.status_code} - {e_resp.error}"  # type: ignore
            raise RuntimeError(self._last_error) from e_resp
        except Exception as e_general:
            if not self._last_error:
                self._last_error = f"Unexpected error in Ollama stream: {type(e_general).__name__} - {e_general}"
            raise RuntimeError(self._last_error) from e_general

    def get_available_models(self) -> List[str]:
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library not installed."
            return []
        if not self.is_configured() or not self._sync_client:
            self._last_error = "OllamaAdapter not configured. Cannot list models."
            return []

        model_names: List[str] = []
        try:
            models_response_data = self._sync_client.list()  # type: ignore
            items_to_parse = []
            if isinstance(models_response_data, dict) and 'models' in models_response_data:
                items_to_parse = models_response_data['models']
            elif isinstance(models_response_data, list):
                items_to_parse = models_response_data
            elif _ollama_types_imported_successfully and OllamaListResponseType is not None and isinstance(
                    models_response_data, OllamaListResponseType):  # type: ignore
                if hasattr(models_response_data, 'models') and isinstance(models_response_data.models,
                                                                          list):  # type: ignore
                    items_to_parse = models_response_data.models  # type: ignore
            elif _ollama_module_present and hasattr(models_response_data,
                                                    '__module__') and models_response_data.__module__.startswith(
                'ollama._types') and type(models_response_data).__name__ == 'ListResponse':  # type: ignore
                if hasattr(models_response_data, 'models') and isinstance(models_response_data.models,
                                                                          list):  # type: ignore
                    items_to_parse = models_response_data.models  # type: ignore

            for item in items_to_parse:
                model_id_to_add = None
                if isinstance(item, dict):
                    model_id_to_add = item.get('name') or item.get('model')
                elif _ollama_types_imported_successfully and OllamaModelType is not None and isinstance(item,
                                                                                                        OllamaModelType):  # type: ignore
                    model_id_to_add = getattr(item, 'name', None) or getattr(item, 'model', None)
                elif _ollama_module_present and hasattr(item, '__module__') and item.__module__.startswith(
                        'ollama._types') and type(item).__name__ == 'Model':  # type: ignore
                    model_id_to_add = getattr(item, 'name', None) or getattr(item, 'model', None)
                elif hasattr(item, 'name') and isinstance(item.name, str):
                    model_id_to_add = item.name  # type: ignore
                elif hasattr(item, 'model') and isinstance(item.model, str):
                    model_id_to_add = item.model  # type: ignore

                if model_id_to_add and isinstance(model_id_to_add, str): model_names.append(model_id_to_add)

            return sorted(list(set(model_names)))
        except ollama.RequestError as e_req:  # type: ignore
            self._last_error = f"Ollama API Request Error (listing models): {e_req}. Is Ollama running at {self._ollama_host}?"
            return []
        except Exception as e:
            self._last_error = f"Unexpected error fetching models from Ollama: {type(e).__name__} - {e}"
            return []

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:  # type: ignore
        ollama_messages: List[Dict[str, Any]] = []
        if self._system_prompt:
            ollama_messages.append({"role": "system", "content": self._system_prompt})

        for msg in history:  # type: ignore
            role_for_api: Optional[str] = None
            if msg.role == USER_ROLE:
                role_for_api = "user"
            elif msg.role == MODEL_ROLE:
                role_for_api = "assistant"
            elif msg.role == SYSTEM_ROLE and not self._system_prompt:
                role_for_api = "system"
            elif msg.role == SYSTEM_ROLE and self._system_prompt:
                continue
            elif msg.role == ERROR_ROLE or (
                    hasattr(msg, 'metadata') and msg.metadata and msg.metadata.get("is_internal")):
                continue  # type: ignore
            else:
                continue

            message_payload: Dict[str, Any] = {"role": role_for_api}
            text_content_for_api = msg.text.strip() if hasattr(msg, 'text') and msg.text else ""  # type: ignore

            if text_content_for_api: message_payload["content"] = text_content_for_api

            if hasattr(msg, 'has_images') and msg.has_images and hasattr(msg,
                                                                         'image_parts') and msg.image_parts:  # type: ignore
                base64_image_list: List[str] = []
                for img_part_dict in msg.image_parts:  # type: ignore
                    if isinstance(img_part_dict, dict) and img_part_dict.get("type") == "image" and isinstance(
                            img_part_dict.get("data"), str):
                        try:
                            base64.b64decode(img_part_dict["data"], validate=True)
                            base64_image_list.append(img_part_dict["data"])
                        except Exception:
                            pass
                if base64_image_list: message_payload["images"] = base64_image_list

            if "content" in message_payload or "images" in message_payload:
                ollama_messages.append(message_payload)
            elif role_for_api == "system" and "content" not in message_payload and "images" not in message_payload:
                ollama_messages.append(message_payload)
        return ollama_messages

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None