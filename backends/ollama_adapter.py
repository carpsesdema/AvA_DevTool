# backends/ollama_adapter.py
import asyncio
import base64
import logging
import sys
import re
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
    DEFAULT_MODEL = "qwen2.5-coder:14b"

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
        effective_model_name = model_name if model_name and model_name.strip() else self.DEFAULT_MODEL
        logger.info(
            f"OllamaAdapter: Configuring. Host: {self._ollama_host}, Effective Model: {effective_model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")

        self._sync_client = None
        self._is_configured = False  # Default to false, set true after successful param setup
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library ('ollama') not installed."
            logger.error(self._last_error)
            return False

        self._model_name = effective_model_name
        self._system_prompt = system_prompt.strip() if isinstance(system_prompt,
                                                                  str) and system_prompt.strip() else None

        try:
            logger.info(f"  Attempting to create Ollama client for host: {self._ollama_host} with timeout 20s.")
            self._sync_client = ollama.Client(host=self._ollama_host, timeout=20.0)
            # Parameters are set, so basic configuration is complete.
            self._is_configured = True
            logger.info(f"  OllamaAdapter parameters configured for model '{self._model_name}' at {self._ollama_host}.")

            # Now, test the connection. Failure here is a warning, not a config failure.
            logger.info(f"  Testing connection to Ollama server with .list() call...")
            try:
                self._sync_client.list()  # Test connection
                logger.info(f"  Successfully connected to Ollama and listed models at {self._ollama_host}.")
            except ollama.RequestError as conn_err_req:
                self._last_error = f"Warning: Failed to connect/list models from Ollama at {self._ollama_host} during initial configure. Is server running? (RequestError): {conn_err_req}"
                logger.warning(self._last_error, exc_info=False)
                # Keep _is_configured = True, as parameters are set. Error will be caught later.
            except ConnectionRefusedError:
                self._last_error = f"Warning: Connection refused by Ollama at {self._ollama_host} during initial configure. Is server running?"
                logger.warning(self._last_error, exc_info=False)
            except Exception as conn_err_other:
                self._last_error = f"Warning: Unexpected error during Ollama connection test at {self._ollama_host}: {type(conn_err_other).__name__} - {conn_err_other}"
                logger.warning(self._last_error, exc_info=True)

            return True  # Configuration of parameters was successful

        except Exception as e:
            self._last_error = f"Unexpected error creating Ollama client for host '{self._ollama_host}': {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)
            self._sync_client = None
            self._is_configured = False  # Critical client creation error means not configured
            return False

    def is_configured(self) -> bool:
        return self._is_configured  # Now primarily reflects if parameters are set

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> \
            AsyncGenerator[str, None]:
        logger.info(
            f"OllamaAdapter: Generating stream. Model: {self._model_name}, History items: {len(history)}, Options: {options}")
        self._last_error = None  # Clear previous errors for this attempt
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._sync_client:
            self._last_error = "Adapter is not configured or client is missing (cannot make LLM call)."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        messages_for_api = self._format_history_for_api(history)
        if not messages_for_api and not self._system_prompt:
            if not messages_for_api and self._system_prompt and not any(
                    m['role'] == 'system' for m in messages_for_api):
                logger.debug("OllamaAdapter: History is empty, but a system prompt is configured. This is valid.")
            else:
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
                # This is where a connection error would manifest if the server is down now
                return self._sync_client.chat(  # type: ignore
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
                if not chunk_error and hasattr(chunk_data_obj, 'message') and chunk_data_obj.message:  # type: ignore
                    chunk_error = getattr(chunk_data_obj.message, 'error', None)  # type: ignore

                if chunk_error:
                    self._last_error = f"Ollama Stream Error: {chunk_error}"
                    logger.error(self._last_error)
                    yield f"[SYSTEM ERROR: {self._last_error}]"
                    if getattr(chunk_data_obj, 'done', False):
                        self._last_prompt_tokens = getattr(chunk_data_obj, 'prompt_eval_count', None)
                        self._last_completion_tokens = getattr(chunk_data_obj, 'eval_count', None)
                    return

                content_part = ''
                if hasattr(chunk_data_obj, 'message') and chunk_data_obj.message:  # type: ignore
                    if hasattr(chunk_data_obj.message, 'content') and chunk_data_obj.message.content:  # type: ignore
                        content_part = chunk_data_obj.message.content  # type: ignore

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
        except ollama.ResponseError as e_resp:  # type: ignore
            self._last_error = f"Ollama API Response Error: {e_resp.status_code} - {e_resp.error}"
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from e_resp
        except ollama.RequestError as e_req:  # type: ignore
            self._last_error = f"Ollama API Request Error (e.g. server down): {e_req}"
            logger.error(self._last_error, exc_info=True)  # Don't print full trace for common conn errors
            raise RuntimeError(self._last_error) from e_req
        except Exception as e_general:
            if not self._last_error:
                self._last_error = f"Unexpected error in Ollama stream ('{self._model_name}'): {type(e_general).__name__} - {e_general}"
            logger.error(self._last_error, exc_info=True)
            if not isinstance(e_general, RuntimeError):
                raise RuntimeError(self._last_error) from e_general
            else:
                raise

    def get_available_models(self) -> List[str]:
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library not installed."
            logger.error(self._last_error)
            return []

        model_names: List[str] = []
        client_to_use: Optional[ollama.Client] = self._sync_client  # type: ignore
        can_attempt_live_fetch = True

        if not client_to_use and self._is_configured:  # Adapter params are set, but client might be None due to earlier creation error
            logger.warning(
                "OllamaAdapter.get_available_models: Adapter is configured but client is None. Attempting to re-create.")
            try:
                client_to_use = ollama.Client(host=self._ollama_host, timeout=7.0)  # type: ignore
                # No need to list here, the thread below will do it.
            except Exception as e_recreate:
                logger.error(f"Failed to re-create Ollama client during get_available_models: {e_recreate}")
                self._last_error = f"Client re-creation failed: {e_recreate}"
                can_attempt_live_fetch = False
        elif not client_to_use:  # Not configured, or client creation failed earlier and _is_configured is False
            logger.info(
                "OllamaAdapter.get_available_models: No pre-configured client. Attempting temporary client for model listing.")
            try:
                import socket
                from urllib.parse import urlparse
                parsed_host = urlparse(
                    self._ollama_host if self._ollama_host.startswith('http') else f'http://{self._ollama_host}')
                host = parsed_host.hostname or 'localhost'
                port = parsed_host.port or 11434
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)  # Quick check
                result = sock.connect_ex((host, port))
                sock.close()
                if result != 0:
                    self._last_error = f"Ollama server not responding at {host}:{port} (socket check failed for model list)."
                    logger.warning(self._last_error)
                    can_attempt_live_fetch = False
                else:  # Socket check passed, try creating client
                    client_to_use = ollama.Client(host=self._ollama_host, timeout=7.0)  # type: ignore
            except Exception as e_temp_client:
                self._last_error = f"Failed to connect temporary Ollama client for model listing: {type(e_temp_client).__name__}"
                logger.warning(self._last_error)
                can_attempt_live_fetch = False

        if not client_to_use:  # If client_to_use is still None
            can_attempt_live_fetch = False

        if can_attempt_live_fetch and client_to_use:
            try:
                import threading
                import queue
                result_queue = queue.Queue()
                exception_queue = queue.Queue()

                def fetch_models_thread():
                    try:
                        models_response = client_to_use.list()  # type: ignore
                        result_queue.put(models_response)
                    except Exception as e_thread:
                        exception_queue.put(e_thread)

                fetch_thread = threading.Thread(target=fetch_models_thread)
                fetch_thread.daemon = True
                fetch_thread.start()
                fetch_thread.join(timeout=10.0)  # Increased timeout slightly
                if fetch_thread.is_alive():
                    logger.warning(f"Ollama model fetch timed out for {self._ollama_host} after 10 seconds.")
                    self._last_error = "Ollama model fetch timed out."
                elif not exception_queue.empty():
                    e_fetch_thread = exception_queue.get()
                    self._last_error = f"Error in Ollama model fetch thread: {type(e_fetch_thread).__name__} - {e_fetch_thread}"
                    logger.warning(self._last_error)  # Log as warning, will fallback to defaults
                elif not result_queue.empty():
                    models_response_data = result_queue.get()
                    items_to_parse = []
                    if isinstance(models_response_data, dict) and 'models' in models_response_data:
                        items_to_parse = models_response_data['models']
                    elif isinstance(models_response_data, list):  # ollama library might return list directly
                        items_to_parse = models_response_data

                    for item in items_to_parse:
                        model_id_to_add = None
                        if _ollama_types_imported_successfully and isinstance(item, _OllamaModelType):
                            model_id_to_add = getattr(item, 'name', None)
                        elif isinstance(item, dict):
                            model_id_to_add = item.get('name') or item.get('model')
                        elif hasattr(item, 'name') and isinstance(item.name, str):  # type: ignore
                            model_id_to_add = item.name  # type: ignore
                        elif hasattr(item, 'model') and isinstance(item.model, str):  # type: ignore
                            model_id_to_add = item.model  # type: ignore

                        if model_id_to_add and isinstance(model_id_to_add, str):
                            model_names.append(model_id_to_add)
                    logger.info(f"Fetched {len(model_names)} models live from Ollama.")
                else:  # No error, no result
                    logger.warning(
                        f"No response or empty model list from Ollama at {self._ollama_host}, but no explicit error.")
                    self._last_error = "Ollama returned no models or an empty list."
            except Exception as e_fetch_outer:  # Catch any other exception during the threaded fetch setup
                self._last_error = f"Outer error during Ollama model fetch: {type(e_fetch_outer).__name__} - {e_fetch_outer}"
                logger.warning(self._last_error, exc_info=True)
        elif not can_attempt_live_fetch:
            logger.info("Skipping live Ollama model fetch due to initial connection/client issues. Will use defaults.")

        # Enhanced default candidates list based on user's provided list
        default_candidates = [
            "qwen2.5-coder:32b", "qwen2.5-coder:14b", "qwen2.5-coder:latest",
            "qwen3:30b", "qwen2:latest", "qwen:latest",
            "devstral:24b", "devstral:latest",
            "codellama:34b", "codellama:13b", "codellama:13b-instruct", "codellama:13b-python", "codellama:7b",
            "codellama:latest",
            "llama3:30b", "llama3:latest",
            "deepseek-coder:6.7b-instruct-q5_K_M", "deepseek-r1:14b",
            "wizardcoder:13b-python",
            "starcoder2:15b-instruct", "starcoder2:7b",
            "mistral:latest",
            "phi3:latest",
            self.DEFAULT_MODEL  # Ensure the adapter's own default is present
        ]
        if self._model_name and self._model_name not in default_candidates:  # Add currently configured model if not in defaults
            default_candidates.insert(0, self._model_name)

        # Combine fetched models with defaults, ensuring uniqueness
        combined_list = list(set(model_names + [m for m in default_candidates if m not in model_names]))
        if not combined_list and self._model_name:  # If fetch failed and defaults are empty for some reason, use current
            combined_list.append(self._model_name)
        elif not combined_list:  # Absolute fallback
            combined_list.append(self.DEFAULT_MODEL)

        combined_list = list(set(combined_list))  # Ensure uniqueness again

        def sort_key_ollama(model_name_str: str):
            name_lower = model_name_str.lower()
            is_latest = "latest" in name_lower
            is_current_adapter_model = model_name_str == self._model_name

            priority = 10  # Default priority

            # Prioritize based on keywords
            if "qwen2.5-coder:32b" in name_lower:
                priority = 0
            elif "qwen2.5-coder:14b" in name_lower:
                priority = 1
            elif "qwen2.5-coder:latest" in name_lower:
                priority = 2
            elif "qwen3:30b" in name_lower:
                priority = 3
            elif "qwen" in name_lower:
                priority = 4
            elif "devstral" in name_lower:
                priority = 5
            elif "codellama" in name_lower:
                priority = 6
            elif "llama3" in name_lower:
                priority = 7
            elif "deepseek" in name_lower or "wizardcoder" in name_lower or "starcoder" in name_lower:
                priority = 8
            elif "mistral" in name_lower or "phi3" in name_lower:
                priority = 9

            size_priority = 100  # Default if no size detected
            if 'b' in name_lower:
                try:
                    size_match = re.search(r'(\d+(?:\.\d+)?)(?:b|B)', name_lower)
                    if size_match:
                        size_priority = -float(size_match.group(1))  # Negative for descending size
                except:
                    pass  # Ignore parsing errors for size

            return (
                0 if is_current_adapter_model else 1,  # Current model first
                priority,  # Keyword priority
                size_priority,  # Size (larger first)
                0 if is_latest else 1,  # "latest" tags
                name_lower  # Alphabetical as final tie-breaker
            )

        final_model_list = sorted(combined_list, key=sort_key_ollama)

        if not final_model_list:  # Should not happen with fallbacks
            logger.error("OllamaAdapter: Final model list is unexpectedly empty. Returning default model only.")
            return [self.DEFAULT_MODEL]

        logger.info(
            f"OllamaAdapter: Returning {len(final_model_list)} available models. Top models: {final_model_list[:5]}")
        return final_model_list

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
            elif msg.role == SYSTEM_ROLE and not self._system_prompt:  # Only add if adapter doesn't have its own system prompt
                role_for_api = "system"
            elif msg.role == SYSTEM_ROLE and self._system_prompt:  # Adapter system prompt takes precedence
                continue
            elif msg.role == ERROR_ROLE or (
                    hasattr(msg, 'metadata') and msg.metadata and msg.metadata.get("is_internal")):
                continue  # Skip error messages or internal system notes
            else:
                logger.warning(f"OllamaAdapter: Skipping message with unhandled role: {msg.role}")
                continue

            message_payload: Dict[str, Any] = {"role": role_for_api}
            text_content_for_api = msg.text.strip() if hasattr(msg, 'text') and msg.text else ""

            if text_content_for_api:
                message_payload["content"] = text_content_for_api
            elif not text_content_for_api and role_for_api == "system" and not self._system_prompt:
                # If it's a system message from history (and adapter has no system prompt), and it's empty, send empty.
                message_payload["content"] = ""

            # Image handling for Ollama (if model supports it, e.g., LLaVA, BakLLaVA)
            if hasattr(msg, 'has_images') and msg.has_images and hasattr(msg, 'image_parts') and msg.image_parts:
                base64_image_list: List[str] = []
                for img_part_dict in msg.image_parts:
                    if isinstance(img_part_dict, dict) and img_part_dict.get("type") == "image" and isinstance(
                            img_part_dict.get("data"), str):
                        try:
                            # Ensure data is not empty/whitespace before adding
                            if len(img_part_dict["data"]) > 10 and not img_part_dict["data"].isspace():
                                base64_image_list.append(img_part_dict["data"])
                            else:
                                logger.warning("Skipping potentially invalid or empty base64 image data.")
                        except Exception as e_b64_check:
                            logger.warning(f"Error checking base64 data in image part: {e_b64_check}")
                if base64_image_list:
                    message_payload["images"] = base64_image_list  # Ollama expects list of base64 strings

            # Add message to API history if it has content or images
            if "content" in message_payload or "images" in message_payload:
                ollama_messages.append(message_payload)
            elif role_for_api == "system" and not self._system_prompt and "content" not in message_payload:
                # This handles the case where a system message from history might be role-only
                # and the adapter itself doesn't have a system prompt.
                logger.debug(
                    f"OllamaAdapter: Adding system message (role only) for {self._model_name} because no adapter system prompt is set.")
                message_payload["content"] = ""  # Ensure content field exists even if empty for system
                ollama_messages.append(message_payload)

        return ollama_messages

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None