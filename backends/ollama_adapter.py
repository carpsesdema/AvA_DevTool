# backends/ollama_adapter.py
import asyncio
import base64
import logging
import sys
import requests
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
        """Get available models with increased timeout and better error handling"""
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library not installed."
            logger.error(self._last_error)
            return []

        model_names: List[str] = []
        client_to_use: Optional[ollama.Client] = self._sync_client

        # Create or get client
        if not client_to_use:
            try:
                client_to_use = ollama.Client(host=self._ollama_host, timeout=30.0)  # Increased timeout
                logger.info(f"Created temporary Ollama client for model fetching: {self._ollama_host}")
            except Exception as e_temp_client:
                self._last_error = f"Failed to create Ollama client: {type(e_temp_client).__name__}"
                logger.error(self._last_error)
                # Return defaults if can't connect
                return self._get_default_model_list()

        # Test connection first
        try:
            # Quick ping test using requests
            response = requests.get(f"{self._ollama_host}/api/tags", timeout=10)
            if response.status_code != 200:
                logger.warning(f"Ollama API returned status {response.status_code}")
                return self._get_default_model_list()

            # Parse response directly
            data = response.json()
            if 'models' in data:
                for model_item in data['models']:
                    if isinstance(model_item, dict) and 'name' in model_item:
                        model_names.append(model_item['name'])
                logger.info(f"Successfully fetched {len(model_names)} models from Ollama API")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Direct API call failed: {e}, trying ollama library...")

            # Fallback to ollama library with threading and longer timeout
            try:
                import threading
                import queue
                result_queue = queue.Queue()
                exception_queue = queue.Queue()

                def fetch_models_thread():
                    try:
                        models_response = client_to_use.list()
                        result_queue.put(models_response)
                    except Exception as e_thread:
                        exception_queue.put(e_thread)

                fetch_thread = threading.Thread(target=fetch_models_thread)
                fetch_thread.daemon = True
                fetch_thread.start()
                fetch_thread.join(timeout=20.0)  # Increased from 10 to 20 seconds

                if fetch_thread.is_alive():
                    logger.warning(f"Ollama model fetch timed out after 20 seconds")
                    self._last_error = "Model fetch timed out"
                elif not exception_queue.empty():
                    e_fetch_thread = exception_queue.get()
                    self._last_error = f"Error fetching models: {type(e_fetch_thread).__name__} - {e_fetch_thread}"
                    logger.warning(self._last_error)
                elif not result_queue.empty():
                    models_response_data = result_queue.get()
                    items_to_parse = []
                    if isinstance(models_response_data, dict) and 'models' in models_response_data:
                        items_to_parse = models_response_data['models']
                    elif isinstance(models_response_data, list):
                        items_to_parse = models_response_data

                    for item in items_to_parse:
                        model_id_to_add = None
                        if isinstance(item, dict):
                            model_id_to_add = item.get('name') or item.get('model')
                        elif hasattr(item, 'name') and isinstance(item.name, str):
                            model_id_to_add = item.name
                        elif hasattr(item, 'model') and isinstance(item.model, str):
                            model_id_to_add = item.model

                        if model_id_to_add and isinstance(model_id_to_add, str):
                            model_names.append(model_id_to_add)

                    logger.info(f"Fetched {len(model_names)} models using ollama library")

            except Exception as e_fetch_outer:
                self._last_error = f"Both API methods failed: {type(e_fetch_outer).__name__} - {e_fetch_outer}"
                logger.error(self._last_error)

        # If we got models, combine with defaults, otherwise just use defaults
        if model_names:
            logger.info(f"Successfully fetched {len(model_names)} models from Ollama")
            combined_list = list(set(model_names + self._get_default_model_list()))
        else:
            logger.warning("No models fetched from Ollama, using defaults only")
            combined_list = self._get_default_model_list()

        # Sort the final list
        final_model_list = sorted(combined_list, key=self._sort_key_ollama)

        # Ensure current model is included
        if self._model_name and self._is_configured and self._model_name not in final_model_list:
            final_model_list.insert(0, self._model_name)
            final_model_list = sorted(final_model_list, key=self._sort_key_ollama)

        logger.info(f"Returning {len(final_model_list)} total models. First 5: {final_model_list[:5]}")
        return final_model_list

    def _get_default_model_list(self) -> List[str]:
        """Get the default model list as fallback"""
        return [
            "qwen2.5-coder:32b", "qwen2.5-coder:14b", "qwen2.5-coder:latest",
            "qwen3:30b", "qwen2:latest", "qwen:latest",
            "devstral:24b", "devstral:latest",
            "codellama:34b", "codellama:13b", "codellama:13b-instruct",
            "codellama:13b-python", "codellama:7b", "codellama:latest",
            "llama3:30b", "llama3:latest",
            "deepseek-coder:6.7b-instruct-q5_K_M", "deepseek-r1:14b",
            "wizardcoder:13b-python",
            "starcoder2:15b-instruct", "starcoder2:7b",
            "mistral:latest", "phi3:latest",
            self.DEFAULT_MODEL
        ]

    def _sort_key_ollama(self, model_name_str: str):
        """Sorting key for Ollama models"""
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