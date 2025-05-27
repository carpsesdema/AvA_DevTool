# backends/ollama_adapter.py
import asyncio
import base64
import logging
import sys
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple, Type

try:
    import ollama

    _ollama_module_present = True
    try:
        from ollama._types import Model as _OllamaModelType
        from ollama._types import ChatResponse as _OllamaChatResponseType

        _ollama_types_imported_successfully = True
    except ImportError:
        _OllamaModelType = object
        _OllamaChatResponseType = object
        _ollama_types_imported_successfully = False
        logging.warning("OllamaAdapter: Could not import specific types from ollama._types. Using generic fallbacks.")
    API_LIBRARY_AVAILABLE = True
except ImportError:
    ollama = None
    _ollama_module_present = False
    _OllamaModelType = object
    _OllamaChatResponseType = object
    _ollama_types_imported_successfully = False
    API_LIBRARY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "OllamaAdapter: 'ollama' library not found. Please install it: pip install ollama")

try:
    from backends.backend_interface import BackendInterface
    from core.models import ChatMessage, MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE
except ImportError:
    BackendInterface = type("BackendInterface", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})
    MODEL_ROLE, USER_ROLE, SYSTEM_ROLE, ERROR_ROLE = "model", "user", "system", "error"

logger = logging.getLogger(__name__)


class OllamaAdapter(BackendInterface):
    DEFAULT_OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_MODEL = "llama3:latest"

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
        logger.info(
            f"OllamaAdapter: Configuring. Host: {self._ollama_host}, Model: {model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")
        self._sync_client = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library ('ollama') not installed."
            logger.error(self._last_error)
            return False

        self._model_name = model_name if model_name and model_name.strip() else self.DEFAULT_MODEL
        self._system_prompt = system_prompt.strip() if isinstance(system_prompt,
                                                                  str) and system_prompt.strip() else None

        try:
            self._sync_client = ollama.Client(host=self._ollama_host, timeout=10.0)  # General client timeout
            try:
                self._sync_client.list()  # Test connection
                logger.info(f"  Successfully connected to Ollama at {self._ollama_host}.")
            except ollama.RequestError as conn_err_req:  # Covers httpx.ConnectError, httpx.ReadTimeout etc.
                self._last_error = f"Failed to connect to Ollama at {self._ollama_host} (RequestError): {conn_err_req}"
                logger.error(self._last_error, exc_info=False)  # Keep exc_info brief for connection errors
                self._sync_client = None
                return False
            except ConnectionRefusedError:  # Explicitly catch this
                self._last_error = f"Connection refused by Ollama at {self._ollama_host}. Is Ollama running?"
                logger.error(self._last_error, exc_info=False)
                self._sync_client = None
                return False
            except Exception as conn_err_other:  # Catch other potential issues
                self._last_error = f"Failed to connect/verify Ollama at {self._ollama_host}: {type(conn_err_other).__name__} - {conn_err_other}"
                logger.error(self._last_error, exc_info=True)
                self._sync_client = None
                return False

            self._is_configured = True
            logger.info(
                f"  OllamaAdapter configured successfully for model '{self._model_name}' at {self._ollama_host}.")
            return True
        except Exception as e:
            self._last_error = f"Unexpected error configuring Ollama client: {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)
            self._sync_client = None
            return False

    def is_configured(self) -> bool:
        return self._is_configured and self._sync_client is not None

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> \
            AsyncGenerator[str, None]:
        logger.info(
            f"OllamaAdapter: Generating stream. Model: {self._model_name}, History items: {len(history)}, Options: {options}")
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._sync_client:
            self._last_error = "Adapter is not configured."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        messages_for_api = self._format_history_for_api(history)
        if not messages_for_api and not self._system_prompt:
            self._last_error = "Cannot send request: No valid messages in history for API format and no system prompt."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        logger.debug(f"  Sending {len(messages_for_api)} formatted messages to model '{self._model_name}'.")

        ollama_api_options: Dict[str, Any] = {}
        if options:
            if "temperature" in options and isinstance(options["temperature"], (float, int)):
                ollama_api_options["temperature"] = float(options["temperature"])
                logger.info(f"  Applying temperature: {ollama_api_options['temperature']} to Ollama request.")

        ollama_sync_iterator = None
        try:
            def _get_ollama_iterator():
                return self._sync_client.chat(
                    model=self._model_name,
                    messages=messages_for_api,
                    stream=True,
                    options=ollama_api_options
                )

            ollama_sync_iterator = await asyncio.to_thread(_get_ollama_iterator)

            loop = asyncio.get_running_loop()
            chunk_count = 0

            while True:
                chunk_data_obj: Optional[_OllamaChatResponseType] = None
                try:
                    chunk_data_obj = await loop.run_in_executor(None, next, ollama_sync_iterator)
                    chunk_count += 1
                except StopIteration:
                    logger.info(
                        f"Ollama stream for '{self._model_name}' ended (StopIteration). Total Chunks: {chunk_count}.")
                    break
                except Exception as e_next:
                    self._last_error = f"Error calling next() on Ollama stream for '{self._model_name}': {type(e_next).__name__} - {e_next}"
                    logger.error(self._last_error, exc_info=True)
                    yield f"[SYSTEM ERROR: {self._last_error}]"
                    return

                logger.debug(
                    f"Ollama raw chunk obj #{chunk_count} for {self._model_name}: Type {type(chunk_data_obj)}, Content: {str(chunk_data_obj)[:250]}")

                chunk_error = getattr(chunk_data_obj, 'error', None)
                if not chunk_error and hasattr(chunk_data_obj, 'message') and chunk_data_obj.message:
                    chunk_error = getattr(chunk_data_obj.message, 'error', None)

                if chunk_error:
                    self._last_error = f"Ollama Stream Error: {chunk_error}"
                    logger.error(self._last_error)
                    yield f"[SYSTEM ERROR: {self._last_error}]"
                    if getattr(chunk_data_obj, 'done', False):
                        self._last_prompt_tokens = getattr(chunk_data_obj, 'prompt_eval_count', None)
                        self._last_completion_tokens = getattr(chunk_data_obj, 'eval_count', None)
                    return

                content_part = ''
                if hasattr(chunk_data_obj, 'message') and chunk_data_obj.message:
                    if hasattr(chunk_data_obj.message, 'content') and chunk_data_obj.message.content:
                        content_part = chunk_data_obj.message.content

                if content_part:
                    yield content_part

                is_done = getattr(chunk_data_obj, 'done', False)
                if is_done:
                    logger.info(f"Ollama stream for '{self._model_name}' finished (done flag). Chunks: {chunk_count}.")
                    self._last_prompt_tokens = getattr(chunk_data_obj, 'prompt_eval_count', None)
                    self._last_completion_tokens = getattr(chunk_data_obj, 'eval_count', None)
                    logger.info(
                        f"  Ollama Token Usage ('{self._model_name}'): Prompt={self._last_prompt_tokens}, Completion={self._last_completion_tokens}")
                    break

        except ollama.ResponseError as e_resp:
            self._last_error = f"Ollama API Response Error: {e_resp.status_code} - {e_resp.error}"
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from e_resp
        except Exception as e_general:
            if not self._last_error:
                self._last_error = f"Unexpected error in Ollama stream ('{self._model_name}'): {type(e_general).__name__} - {e_general}"
            logger.error(self._last_error, exc_info=True)
            if not isinstance(e_general, RuntimeError):
                raise RuntimeError(self._last_error) from e_general
            else:
                raise

    def get_available_models(self) -> List[str]:
        """Get available models with timeout protection"""
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library not installed."
            logger.error(self._last_error)
            return []

        client_to_use: Optional[ollama.Client] = self._sync_client

        if not client_to_use:
            try:
                logger.info("OllamaAdapter.get_available_models: Creating temporary client with short timeout")
                # NEW: Much shorter timeout for startup checks
                client_to_use = ollama.Client(host=self._ollama_host, timeout=2.0)

                # NEW: Quick connection test with socket check
                import socket
                from urllib.parse import urlparse

                # Parse host to check if Ollama is running
                parsed = urlparse(
                    self._ollama_host if self._ollama_host.startswith('http') else f'http://{self._ollama_host}')
                host = parsed.hostname or 'localhost'
                port = parsed.port or 11434

                # Quick socket check before API call
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)  # 1 second timeout
                result = sock.connect_ex((host, port))
                sock.close()

                if result != 0:
                    self._last_error = f"Ollama not running on {self._ollama_host}"
                    logger.warning(self._last_error)
                    return []

                # Quick test to see if server is reachable
                client_to_use.list()

            except socket.timeout:
                self._last_error = f"Ollama connection timeout to {self._ollama_host}"
                logger.warning(self._last_error)
                return []
            except Exception as e_temp_client:
                self._last_error = f"Ollama not reachable at {self._ollama_host}: {type(e_temp_client).__name__}"
                logger.warning(self._last_error)
                return []

        if not client_to_use:
            self._last_error = "Ollama client not available for listing models."
            logger.error(self._last_error)
            return []

        model_names: List[str] = []
        try:
            # NEW: Use threading for timeout protection
            import threading
            import queue

            result_queue = queue.Queue()
            exception_queue = queue.Queue()

            def fetch_models():
                try:
                    models_response = client_to_use.list()
                    result_queue.put(models_response)
                except Exception as e:
                    exception_queue.put(e)

            # Start fetch in separate thread
            fetch_thread = threading.Thread(target=fetch_models)
            fetch_thread.daemon = True
            fetch_thread.start()

            # Wait with timeout
            fetch_thread.join(timeout=3.0)  # 3 second max wait

            if fetch_thread.is_alive():
                logger.warning(f"Ollama model fetch timed out for {self._ollama_host}")
                return self.DEFAULT_MODEL  # Return at least the default

            # Check for exceptions
            if not exception_queue.empty():
                raise exception_queue.get()

            # Get results
            if result_queue.empty():
                logger.warning(f"No response from Ollama at {self._ollama_host}")
                return [self.DEFAULT_MODEL]

            models_response_data = result_queue.get()

            # Parse models (keeping your existing parsing logic)
            items_to_parse = []
            if isinstance(models_response_data, dict) and 'models' in models_response_data:
                items_to_parse = models_response_data['models']
            elif isinstance(models_response_data, list):
                items_to_parse = models_response_data

            for item in items_to_parse:
                model_id_to_add = None
                if _ollama_types_imported_successfully and isinstance(item, _OllamaModelType):
                    model_id_to_add = getattr(item, 'name', None)
                elif isinstance(item, dict):
                    model_id_to_add = item.get('name') or item.get('model')
                elif hasattr(item, 'name') and isinstance(item.name, str):
                    model_id_to_add = item.name
                elif hasattr(item, 'model') and isinstance(item.model, str):
                    model_id_to_add = item.model

                if model_id_to_add and isinstance(model_id_to_add, str):
                    model_names.append(model_id_to_add)

            # Add defaults if no models found
            if not model_names:
                model_names.append(self.DEFAULT_MODEL)

            # Keep your existing sorting logic
            default_candidates = ["llama3:latest", "codellama:latest", "mistral:latest", "phi3:latest", "qwen2:latest",
                                  self.DEFAULT_MODEL]
            if self._model_name and self._model_name != self.DEFAULT_MODEL:
                default_candidates.append(self._model_name)

            combined_list = list(set(model_names + [m for m in default_candidates if m not in model_names]))

            def sort_key_ollama(model_name_str: str):
                name_lower = model_name_str.lower()
                is_latest = "latest" in name_lower
                is_default_class_model = model_name_str == self.DEFAULT_MODEL
                is_current_adapter_model = model_name_str == self._model_name

                priority = 5
                if "llama3" in name_lower:
                    priority = 0
                elif "devstral" in name_lower:
                    priority = 0
                elif "qwen" in name_lower:
                    priority = 1
                elif "codellama" in name_lower:
                    priority = 2
                elif "mistral" in name_lower:
                    priority = 3
                elif "phi3" in name_lower:
                    priority = 4

                return (
                    0 if is_current_adapter_model else 1,
                    0 if is_default_class_model else 1,
                    priority,
                    0 if is_latest else 1,
                    name_lower
                )

            final_model_list = sorted(combined_list, key=sort_key_ollama)
            return final_model_list

        except ollama.RequestError as e_req:
            self._last_error = f"Ollama API Request Error: {e_req}. Is server running?"
            logger.warning(self._last_error)
            return [self.DEFAULT_MODEL]  # Return default instead of empty list
        except Exception as e:
            self._last_error = f"Error fetching Ollama models: {type(e).__name__} - {e}"
            logger.warning(self._last_error)
            return [self.DEFAULT_MODEL]  # Return default instead of empty list

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:
        ollama_messages: List[Dict[str, Any]] = []
        if self._system_prompt:
            ollama_messages.append({"role": "system", "content": self._system_prompt})

        for msg in history:
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
                continue
            else:
                logger.warning(f"OllamaAdapter: Skipping message with unhandled role: {msg.role}")
                continue

            message_payload: Dict[str, Any] = {"role": role_for_api}
            text_content_for_api = msg.text.strip() if hasattr(msg, 'text') and msg.text else ""

            if text_content_for_api:
                message_payload["content"] = text_content_for_api

            if hasattr(msg, 'has_images') and msg.has_images and hasattr(msg, 'image_parts') and msg.image_parts:
                base64_image_list: List[str] = []
                for img_part_dict in msg.image_parts:
                    if isinstance(img_part_dict, dict) and img_part_dict.get("type") == "image" and isinstance(
                            img_part_dict.get("data"), str):
                        try:
                            base64.b64decode(img_part_dict["data"], validate=True)
                            base64_image_list.append(img_part_dict["data"])
                        except Exception as e_b64:
                            logger.warning(f"Invalid base64 data in image part. Skipping. Error: {e_b64}")
                if base64_image_list:
                    message_payload["images"] = base64_image_list

            if "content" in message_payload or "images" in message_payload:
                ollama_messages.append(message_payload)
            elif role_for_api == "system":
                logger.debug(f"OllamaAdapter: Adding system message (role only) for {self._model_name}")
                ollama_messages.append(message_payload)
        return ollama_messages

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None