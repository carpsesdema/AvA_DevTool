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
        # Attempt to import specific types for better type checking if available
        from ollama._types import Model as _OllamaModelType  # For model listing
        from ollama._types import ChatResponse as _OllamaChatResponseType  # For chat stream

        _ollama_types_imported_successfully = True
    except ImportError:
        _OllamaModelType = object  # Fallback type
        _OllamaChatResponseType = object  # Fallback type
        _ollama_types_imported_successfully = False
        logging.warning("OllamaAdapter: Could not import specific types from ollama._types. Using generic fallbacks.")
    API_LIBRARY_AVAILABLE = True
except ImportError:
    ollama = None  # type: ignore
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
            self._sync_client = ollama.Client(host=self._ollama_host)
            try:
                self._sync_client.list()
                logger.info(f"  Successfully connected to Ollama at {self._ollama_host}.")
            except ollama.RequestError as conn_err_req:
                self._last_error = f"Failed to connect to Ollama at {self._ollama_host} (RequestError): {conn_err_req}"
                logger.error(self._last_error, exc_info=True)
                self._sync_client = None
                return False
            except ConnectionRefusedError:
                self._last_error = f"Connection refused by Ollama at {self._ollama_host}. Is Ollama running?"
                logger.error(self._last_error, exc_info=True)
                self._sync_client = None
                return False
            except Exception as conn_err_other:
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
            # Get the synchronous stream iterator by running the initial call in a thread
            def _get_ollama_iterator():
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
                chunk_data_obj: Optional[_OllamaChatResponseType] = None  # type: ignore
                try:
                    # Run the blocking next() call in an executor to make it non-blocking for asyncio
                    chunk_data_obj = await loop.run_in_executor(None, next, ollama_sync_iterator)
                    chunk_count += 1
                except StopIteration:
                    logger.info(
                        f"Ollama stream for '{self._model_name}' ended (StopIteration). Total Chunks: {chunk_count}.")
                    break
                except Exception as e_next:  # Catch other errors from next()
                    self._last_error = f"Error calling next() on Ollama stream for '{self._model_name}': {type(e_next).__name__} - {e_next}"
                    logger.error(self._last_error, exc_info=True)
                    yield f"[SYSTEM ERROR: {self._last_error}]"
                    return  # Stop generation

                # Process the chunk_data_obj (which should be an instance of ollama._types.ChatResponse or similar)
                logger.debug(
                    f"Ollama raw chunk obj #{chunk_count} for {self._model_name}: Type {type(chunk_data_obj)}, Content: {str(chunk_data_obj)[:250]}")

                # Access attributes directly from the ChatResponse object
                chunk_error = getattr(chunk_data_obj, 'error', None)  # Some versions might have error directly
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
            else:  # Re-raise original RuntimeError
                raise

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
            models_response_data = self._sync_client.list()
            items_to_parse = []
            # Handle different structures the ollama library might return for list()
            if isinstance(models_response_data, dict) and 'models' in models_response_data:
                items_to_parse = models_response_data['models']
            elif isinstance(models_response_data, list):
                items_to_parse = models_response_data  # Older versions might return a list of model objects/dicts
            # REMOVED THE PROBLEMATIC _OllamaListResponseType CHECK THAT WAS CAUSING THE BUG

            for item in items_to_parse:
                model_id_to_add = None
                if _ollama_types_imported_successfully and isinstance(item, _OllamaModelType):  # type: ignore
                    model_id_to_add = getattr(item, 'name', None)
                elif isinstance(item, dict):
                    model_id_to_add = item.get('name') or item.get('model')  # Common dict keys
                elif hasattr(item, 'name') and isinstance(item.name, str):  # Generic object attribute
                    model_id_to_add = item.name
                elif hasattr(item, 'model') and isinstance(item.model, str):  # Fallback for 'model' attribute
                    model_id_to_add = item.model

                if model_id_to_add and isinstance(model_id_to_add, str):
                    model_names.append(model_id_to_add)
                else:
                    logger.warning(f"Could not extract model name from item: {item} (Type: {type(item)})")

            return sorted(list(set(model_names)))
        except ollama.RequestError as e_req:
            self._last_error = f"Ollama API Request Error (listing models): {e_req}. Is Ollama server running at {self._ollama_host}?"
            logger.error(self._last_error, exc_info=True)
            return []
        except Exception as e:
            self._last_error = f"Unexpected error fetching models from Ollama: {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)
            return []

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
            elif role_for_api == "system":  # Allow system message even if content/images are empty (e.g. role only)
                logger.debug(f"OllamaAdapter: Adding system message (role only) for {self._model_name}")
                ollama_messages.append(message_payload)
        return ollama_messages

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None