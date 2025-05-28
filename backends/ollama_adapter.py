# backends/ollama_adapter.py - Enhanced with NO TIMEOUTS for complex RAG requests
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
        logger.info("Enhanced OllamaAdapter initialized with NO TIMEOUTS for complex requests")

    def configure(self, api_key: Optional[str], model_name: Optional[str], system_prompt: Optional[str] = None) -> bool:
        effective_model_name = model_name if model_name and model_name.strip() else self.DEFAULT_MODEL
        logger.info(
            f"OllamaAdapter: Configuring for INFINITE PATIENCE. Host: {self._ollama_host}, Model: {effective_model_name}")

        self._sync_client = None
        self._is_configured = False
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
            logger.info(f"Creating Ollama client with INFINITE PATIENCE settings...")
            # ENHANCED: Remove timeouts completely for complex RAG requests
            self._sync_client = ollama.Client(host=self._ollama_host)

            self._is_configured = True
            logger.info(f"OllamaAdapter configured for infinite patience with model '{self._model_name}' at {self._ollama_host}")

            # Test connection with very lenient timeout
            logger.info(f"Testing connection to Ollama server (patient mode)...")
            try:
                # Use a much more lenient timeout for connection test
                import threading
                import queue
                result_queue = queue.Queue()
                exception_queue = queue.Queue()

                def test_connection():
                    try:
                        models = self._sync_client.list()
                        result_queue.put(models)
                    except Exception as e:
                        exception_queue.put(e)

                test_thread = threading.Thread(target=test_connection)
                test_thread.daemon = True
                test_thread.start()
                test_thread.join(timeout=30.0)  # Increased to 30 seconds for initial test

                if test_thread.is_alive():
                    self._last_error = f"Connection test taking longer than expected but continuing in patient mode."
                    logger.warning(self._last_error)
                elif not exception_queue.empty():
                    e = exception_queue.get()
                    self._last_error = f"Connection test warning: {e}. Will retry during actual requests."
                    logger.warning(self._last_error)
                else:
                    logger.info(f"Successfully connected to Ollama at {self._ollama_host} in patient mode.")

            except Exception as conn_err:
                self._last_error = f"Connection test warning: {conn_err}. Will proceed with patient retries."
                logger.warning(self._last_error)

            return True

        except Exception as e:
            self._last_error = f"Failed to create Ollama client: {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)
            self._sync_client = None
            self._is_configured = False
            return False

    def is_configured(self) -> bool:
        return self._is_configured

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> \
    AsyncGenerator[str, None]:
        logger.info(
            f"OllamaAdapter: Starting PATIENT streaming with NO TIMEOUTS. Model: {self._model_name}, History: {len(history)} items")
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._sync_client:
            self._last_error = "Adapter not configured or client missing."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        messages_for_api = self._format_history_for_api(history)
        if not messages_for_api and not self._system_prompt:
            self._last_error = "No valid messages and no system prompt."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        logger.debug(f"Sending {len(messages_for_api)} formatted messages to model '{self._model_name}' with infinite patience.")

        ollama_api_options: Dict[str, Any] = {}
        if options:
            if "temperature" in options and isinstance(options["temperature"], (float, int)):
                ollama_api_options["temperature"] = float(options["temperature"])

        # ENHANCED: Use patient streaming with NO TIMEOUTS
        try:
            async for chunk in self._stream_with_infinite_patience(messages_for_api, ollama_api_options):
                yield chunk
        except Exception as e:
            error_msg = f"Patient streaming error: {type(e).__name__} - {e}"
            self._last_error = error_msg
            logger.error(error_msg, exc_info=True)
            yield f"[SYSTEM ERROR: {error_msg}]"

    async def _stream_with_infinite_patience(self, messages: List[Dict[str, Any]], options: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """ENHANCED: Streaming with infinite patience and NO TIMEOUTS for complex RAG requests."""

        max_retries = 5  # Increased retries
        retry_delay = 5.0  # Initial retry delay

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Starting Ollama stream attempt {attempt + 1}/{max_retries + 1} with INFINITE patience")

                start_time = asyncio.get_event_loop().time()
                chunk_count = 0

                # Create the stream with NO TIMEOUT PROTECTION
                try:
                    ollama_sync_iterator = await self._create_ollama_stream_patient(messages, options)
                except Exception as e:
                    logger.warning(f"Stream creation failed on attempt {attempt + 1}: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                        continue
                    raise

                # Process the stream with INFINITE PATIENCE
                loop = asyncio.get_running_loop()

                while True:
                    try:
                        # NO TIMEOUT - wait forever for chunks
                        chunk_future = loop.run_in_executor(None, self._get_next_chunk, ollama_sync_iterator)
                        chunk_data_obj = await chunk_future  # NO TIMEOUT HERE

                        if chunk_data_obj is None:  # Stream ended normally
                            logger.info(f"Ollama stream ended normally after {chunk_count} chunks")
                            return

                        # Process the chunk
                        chunk_error = getattr(chunk_data_obj, 'error', None)
                        if chunk_error:
                            error_msg = f"Ollama chunk error: {chunk_error}"
                            self._last_error = error_msg
                            logger.error(error_msg)
                            yield f"[SYSTEM ERROR: {error_msg}]"
                            return

                        # Extract content
                        content_part = ''
                        if hasattr(chunk_data_obj, 'message') and chunk_data_obj.message:
                            if hasattr(chunk_data_obj.message, 'content') and chunk_data_obj.message.content:
                                content_part = chunk_data_obj.message.content

                        if content_part:
                            yield content_part
                            chunk_count += 1

                        # Check if done
                        is_done = getattr(chunk_data_obj, 'done', False)
                        if is_done:
                            logger.info(f"Ollama stream completed successfully with infinite patience. Chunks: {chunk_count}")
                            # Extract token usage
                            self._last_prompt_tokens = getattr(chunk_data_obj, 'prompt_eval_count', None)
                            self._last_completion_tokens = getattr(chunk_data_obj, 'eval_count', None)
                            return

                    except Exception as e:
                        logger.error(f"Error processing chunk on attempt {attempt + 1}: {e}", exc_info=True)
                        if attempt < max_retries:
                            break  # Try next attempt
                        else:
                            raise

                    # Brief pause to prevent tight loop
                    await asyncio.sleep(0.01)

                # If we got here, this attempt failed but we're retrying
                continue

            except Exception as e:
                logger.warning(f"Error on patient attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying with infinite patience in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    logger.error(f"All retry attempts failed even with infinite patience: {e}")
                    raise

    async def _create_ollama_stream_patient(self, messages: List[Dict[str, Any]], options: Dict[str, Any]):
        """Create the Ollama stream iterator with patient settings."""
        loop = asyncio.get_running_loop()

        def _sync_create_stream():
            return self._sync_client.chat(
                model=self._model_name,
                messages=messages,
                stream=True,
                options=options
            )

        try:
            return await loop.run_in_executor(None, _sync_create_stream)
        except Exception as e:
            logger.error(f"Failed to create patient Ollama stream: {e}")
            raise

    def _get_next_chunk(self, stream_iterator):
        """Get the next chunk from stream iterator (synchronous) with patience."""
        try:
            return next(stream_iterator)
        except StopIteration:
            return None
        except Exception as e:
            logger.error(f"Error getting next chunk: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get available models with very patient timeout protection."""
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Ollama library not installed."
            return self._get_default_model_list()

        model_names: List[str] = []

        # Use multiple approaches for robustness
        success = False

        # Approach 1: Direct API call with increased timeout
        try:
            response = requests.get(f"{self._ollama_host}/api/tags", timeout=60)  # Increased to 60 seconds
            if response.status_code == 200:
                data = response.json()
                if 'models' in data:
                    for model_item in data['models']:
                        if isinstance(model_item, dict) and 'name' in model_item:
                            model_names.append(model_item['name'])
                    success = True
                    logger.info(f"Successfully fetched {len(model_names)} models via patient direct API")
        except Exception as e:
            logger.warning(f"Patient direct API approach failed: {e}")

        # Approach 2: Ollama library with very patient timeout (fallback)
        if not success and self._sync_client:
            try:
                import threading
                import queue

                result_queue = queue.Queue()
                exception_queue = queue.Queue()

                def fetch_models_thread():
                    try:
                        models_response = self._sync_client.list()
                        result_queue.put(models_response)
                    except Exception as e_thread:
                        exception_queue.put(e_thread)

                fetch_thread = threading.Thread(target=fetch_models_thread)
                fetch_thread.daemon = True
                fetch_thread.start()
                fetch_thread.join(timeout=60.0)  # Increased to 60 seconds

                if fetch_thread.is_alive():
                    logger.warning("Ollama model fetch taking longer than 60 seconds, using defaults")
                elif not exception_queue.empty():
                    e_fetch = exception_queue.get()
                    logger.warning(f"Error fetching models via patient library: {e_fetch}")
                elif not result_queue.empty():
                    models_response_data = result_queue.get()
                    # Parse response
                    items_to_parse = []
                    if isinstance(models_response_data, dict) and 'models' in models_response_data:
                        items_to_parse = models_response_data['models']
                    elif isinstance(models_response_data, list):
                        items_to_parse = models_response_data

                    for item in items_to_parse:
                        model_id = None
                        if isinstance(item, dict):
                            model_id = item.get('name') or item.get('model')
                        elif hasattr(item, 'name'):
                            model_id = item.name

                        if model_id and isinstance(model_id, str):
                            model_names.append(model_id)

                    success = True
                    logger.info(f"Fetched {len(model_names)} models using patient library")

            except Exception as e_lib:
                logger.warning(f"Patient library approach also failed: {e_lib}")

        # Combine with defaults
        if model_names:
            combined_list = list(set(model_names + self._get_default_model_list()))
        else:
            logger.warning("No models fetched from Ollama even with patience, using defaults only")
            combined_list = self._get_default_model_list()

        # Sort and return
        final_model_list = sorted(combined_list, key=self._sort_key_ollama)

        if self._model_name and self._is_configured and self._model_name not in final_model_list:
            final_model_list.insert(0, self._model_name)
            final_model_list = sorted(final_model_list, key=self._sort_key_ollama)

        logger.info(f"Returning {len(final_model_list)} total models with patient processing")
        return final_model_list

    def _get_default_model_list(self) -> List[str]:
        """Get default model list as fallback."""
        return [
            "qwen2.5-coder:32b", "qwen2.5-coder:14b", "qwen2.5-coder:latest",
            "qwen3:30b", "qwen2:latest", "qwen:latest",
            "devstral:24b", "devstral:latest",
            "codellama:34b", "codellama:13b", "codellama:7b", "codellama:latest",
            "llama3:30b", "llama3:latest",
            "deepseek-coder:6.7b-instruct-q5_K_M", "deepseek-r1:14b",
            "wizardcoder:13b-python",
            "starcoder2:15b-instruct", "starcoder2:7b",
            "mistral:latest", "phi3:latest",
            self.DEFAULT_MODEL
        ]

    def _sort_key_ollama(self, model_name_str: str):
        """Sorting key for Ollama models."""
        name_lower = model_name_str.lower()
        is_latest = "latest" in name_lower
        is_current = model_name_str == self._model_name

        priority = 10
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

        size_priority = 100
        if 'b' in name_lower:
            try:
                size_match = re.search(r'(\d+(?:\.\d+)?)(?:b|B)', name_lower)
                if size_match:
                    size_priority = -float(size_match.group(1))
            except:
                pass

        return (
            0 if is_current else 1,
            priority,
            size_priority,
            0 if is_latest else 1,
            name_lower
        )

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Format history for Ollama API with enhanced error handling."""
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
            text_content = msg.text.strip() if hasattr(msg, 'text') and msg.text else ""

            if text_content:
                message_payload["content"] = text_content
            elif role_for_api == "system" and not self._system_prompt:
                message_payload["content"] = ""

            # Image handling
            if hasattr(msg, 'has_images') and msg.has_images and hasattr(msg, 'image_parts') and msg.image_parts:
                base64_image_list: List[str] = []
                for img_part_dict in msg.image_parts:
                    if isinstance(img_part_dict, dict) and img_part_dict.get("type") == "image":
                        try:
                            if len(img_part_dict.get("data", "")) > 10:
                                base64_image_list.append(img_part_dict["data"])
                        except Exception as e_img:
                            logger.warning(f"Error processing image: {e_img}")

                if base64_image_list:
                    message_payload["images"] = base64_image_list

            if "content" in message_payload or "images" in message_payload:
                ollama_messages.append(message_payload)

        return ollama_messages

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None