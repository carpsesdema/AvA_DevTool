# backends/gemini_adapter.py
import asyncio
import logging
import os
import sys
import base64
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

try:
    from backends.backend_interface import BackendInterface
    from core.models import ChatMessage, MODEL_ROLE, USER_ROLE
except ImportError:
    BackendInterface = type("BackendInterface", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})
    MODEL_ROLE, USER_ROLE = "model", "user"

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory, GenerationConfig
    from google.generativeai.types.generation_types import GenerateContentResponse, BlockedPromptException
    from google.api_core.exceptions import GoogleAPIError, ClientError, PermissionDenied, ResourceExhausted, \
        InvalidArgument

    API_LIBRARY_AVAILABLE = True
except ImportError:
    genai = None
    HarmCategory = type("HarmCategory", (object,), {})
    HarmBlockThreshold = type("HarmBlockThreshold", (object,), {})
    GenerationConfig = type("GenerationConfig", (object,), {})
    GenerateContentResponse = type("GenerateContentResponse", (object,), {})
    BlockedPromptException = type("BlockedPromptException", (Exception,), {})
    GoogleAPIError = type("GoogleAPIError", (Exception,), {})
    ClientError = type("ClientError", (GoogleAPIError,), {})
    PermissionDenied = type("PermissionDenied", (ClientError,), {})
    ResourceExhausted = type("ResourceExhausted", (ClientError,), {})
    InvalidArgument = type("InvalidArgument", (ClientError,), {})
    API_LIBRARY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "GeminiAdapter: google-generativeai library not found. Please install it: pip install google-generativeai")

logger = logging.getLogger(__name__)


class GeminiAdapter(BackendInterface):
    def __init__(self):
        self._model: Optional[genai.GenerativeModel] = None
        self._model_name: Optional[str] = None
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        logger.info("GeminiAdapter initialized.")

    def configure(self,
                  api_key: Optional[str],
                  model_name: str,
                  system_prompt: Optional[str] = None) -> bool:
        logger.info(
            f"GeminiAdapter: Configuring. Model: {model_name}. System Prompt: {'Yes' if system_prompt else 'No'}")
        self._model = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Gemini API library (google-generativeai) not installed."
            logger.error(self._last_error)
            return False

        env_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        effective_api_key = api_key if api_key and api_key.strip() else env_api_key

        if not effective_api_key:
            self._last_error = "Gemini API key not provided and not found in GOOGLE_API_KEY/GEMINI_API_KEY environment variables."
            logger.error(self._last_error)
            return False

        if not model_name:
            self._last_error = "Model name is required for Gemini configuration."
            logger.error(self._last_error)
            return False

        try:
            genai.configure(api_key=effective_api_key)

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            effective_system_instruction = system_prompt.strip() if system_prompt and system_prompt.strip() else None

            self._model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=safety_settings,
                system_instruction=effective_system_instruction
            )
            self._model_name = model_name
            self._system_prompt = effective_system_instruction
            self._is_configured = True
            logger.info(f"  GeminiAdapter configured successfully for model '{self._model_name}'.")
            return True
        except ValueError as ve:
            self._last_error = f"Gemini Configuration Error (ValueError): {ve}."
            logger.error(self._last_error, exc_info=True)
        except InvalidArgument as iae:
            self._last_error = f"Gemini API Error (Invalid Argument): {iae}."
            logger.error(self._last_error, exc_info=True)
        except PermissionDenied as pde:
            self._last_error = f"Gemini API Error (Permission Denied): {pde}. Check API key and model access."
            logger.error(self._last_error, exc_info=True)
        except Exception as e:
            self._last_error = f"Unexpected error configuring Gemini model '{model_name}': {type(e).__name__} - {e}"
            logger.error(self._last_error, exc_info=True)

        self._is_configured = False
        return False

    def is_configured(self) -> bool:
        return self._is_configured and self._model is not None

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self,
                                  history: List[ChatMessage],
                                  options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        logger.info(
            f"GeminiAdapter: Generating stream. Model: {self._model_name}, History items: {len(history)}, Options: {options}")
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._model:
            self._last_error = "Adapter not configured or Gemini model object missing."
            logger.error(self._last_error)
            raise RuntimeError(self._last_error)

        gemini_history_api_format = self._format_history_for_api(history)
        if not gemini_history_api_format and not self._system_prompt:
            self._last_error = "Cannot send request: No valid messages and no system instruction."
            logger.error(self._last_error)
            raise ValueError(self._last_error)

        logger.debug(f"  Sending {len(gemini_history_api_format)} formatted messages to model '{self._model_name}'.")

        generation_config_dict = {}
        if options and "temperature" in options and isinstance(options["temperature"], (float, int)):
            generation_config_dict["temperature"] = float(options["temperature"])
            logger.info(
                f"  Applying temperature from options: {generation_config_dict['temperature']} to Gemini request.")

        effective_generation_config = GenerationConfig(**generation_config_dict) if generation_config_dict else None

        sync_iterable_response: Optional[GenerateContentResponse] = None

        try:
            def _initiate_stream_call_in_thread() -> GenerateContentResponse:
                return self._model.generate_content(
                    contents=gemini_history_api_format,
                    stream=True,
                    generation_config=effective_generation_config
                )

            sync_iterable_response = await asyncio.to_thread(_initiate_stream_call_in_thread)

            # The GenerateContentResponse object (sync_iterable_response) is directly iterable for chunks
            # when stream=True. We iterate it synchronously, but yield from our async generator.
            # To ensure GUI responsiveness, we add a small sleep periodically.

            chunk_count = 0
            if not hasattr(sync_iterable_response, '__iter__'):
                # This case should ideally not be hit if stream=True worked as expected.
                # It implies an error or non-streaming response.
                logger.error(
                    f"Gemini response object is not iterable as expected for model {self._model_name}. Type: {type(sync_iterable_response)}")
                if hasattr(sync_iterable_response, 'prompt_feedback') and sync_iterable_response.prompt_feedback and \
                        hasattr(sync_iterable_response.prompt_feedback,
                                'block_reason') and sync_iterable_response.prompt_feedback.block_reason:  # type: ignore
                    err_msg = f"Content blocked by API: {sync_iterable_response.prompt_feedback.block_reason}."  # type: ignore
                    self._last_error = err_msg
                    yield f"[SYSTEM ERROR: {err_msg}]"
                elif hasattr(sync_iterable_response, 'text'):
                    yield sync_iterable_response.text  # type: ignore
                else:
                    self._last_error = "Gemini API did not return an iterable stream or directly readable response."
                    yield f"[SYSTEM ERROR: {self._last_error}]"
                # Attempt to get usage metadata even from a non-iterable (potentially error) response
                if sync_iterable_response and hasattr(sync_iterable_response,
                                                      'usage_metadata') and sync_iterable_response.usage_metadata:  # type: ignore
                    self._process_usage_metadata(sync_iterable_response.usage_metadata)  # type: ignore
                return

            # Iterate directly over the GenerateContentResponse object
            for chunk in sync_iterable_response:  # type: ignore
                chunk_count += 1
                logger.debug(f"Gemini raw chunk #{chunk_count} for {self._model_name}: Type {type(chunk)}")

                # --- START CHUNK PROCESSING ---
                if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and \
                        hasattr(chunk.prompt_feedback, 'block_reason') and chunk.prompt_feedback.block_reason:
                    err_msg = f"Content blocked by API (chunk feedback): {chunk.prompt_feedback.block_reason}."
                    self._last_error = err_msg;
                    logger.error(self._last_error)
                    yield f"[SYSTEM ERROR: {err_msg}]";
                    return

                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason and \
                                candidate.finish_reason.name not in ["STOP", "MAX_TOKENS", "UNSPECIFIED", "NULL"]:
                            err_msg = f"Generation stopped by API (chunk). Reason: {candidate.finish_reason.name}."
                            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                ratings_str = "; ".join(
                                    [f"{sr.category.name}: {sr.probability.name}" for sr in candidate.safety_ratings])
                                err_msg += f" Safety Details: {ratings_str}"
                            self._last_error = err_msg;
                            logger.error(self._last_error)
                            yield f"[SYSTEM ERROR: {err_msg}]";
                            return

                text_parts_from_chunk = []
                if hasattr(chunk, 'parts') and chunk.parts:
                    for part_item in chunk.parts:
                        if hasattr(part_item, 'text') and part_item.text is not None:
                            text_parts_from_chunk.append(part_item.text)
                elif hasattr(chunk, 'text') and chunk.text is not None:
                    text_parts_from_chunk.append(chunk.text)
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    for cand in chunk.candidates:
                        if hasattr(cand, 'content') and cand.content and \
                                hasattr(cand.content, 'parts') and cand.content.parts:
                            for part_item_cand in cand.content.parts:
                                if hasattr(part_item_cand, 'text') and part_item_cand.text is not None:
                                    text_parts_from_chunk.append(part_item_cand.text)

                if text_parts_from_chunk:
                    yield "".join(text_parts_from_chunk)
                # --- END CHUNK PROCESSING ---

                if chunk_count % 3 == 0:
                    await asyncio.sleep(0)  # Cooperative yield

            logger.info(f"Gemini stream for {self._model_name} iteration completed. Total Chunks: {chunk_count}.")

            # Usage metadata is on the main GenerateContentResponse object
            if sync_iterable_response and hasattr(sync_iterable_response,
                                                  'usage_metadata') and sync_iterable_response.usage_metadata:  # type: ignore
                self._process_usage_metadata(sync_iterable_response.usage_metadata)  # type: ignore
            else:
                logger.warning(
                    f"Gemini usage_metadata not found on the main stream response object for {self._model_name}.")

        except BlockedPromptException as bpe_outer:
            self._last_error = f"Gemini API Error: Prompt blocked before streaming. {bpe_outer.args}"
            logger.error(self._last_error, exc_info=True)
            yield f"[SYSTEM ERROR: {self._last_error}]";
            return
        except InvalidArgument as iae:
            self._last_error = f"Gemini API Error (Invalid Argument): {iae}."
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from iae
        except PermissionDenied as pde:
            self._last_error = f"Gemini API Error (Permission Denied): {pde}."
            logger.error(self._last_error, exc_info=True)
            raise RuntimeError(self._last_error) from pde
        except Exception as e_general:
            if not self._last_error:
                self._last_error = f"Error during Gemini stream processing ({self._model_name}): {type(e_general).__name__} - {e_general}"
            logger.error(self._last_error, exc_info=True)
            if not isinstance(e_general, RuntimeError):  # Avoid double-wrapping RuntimeErrors
                yield f"[SYSTEM ERROR: {self._last_error}]"
            else:  # Re-raise if it's already a RuntimeError we want to propagate
                raise
            return

    def _process_usage_metadata(self, usage_metadata_obj):
        """Helper to process usage_metadata."""
        if not usage_metadata_obj: return

        self._last_prompt_tokens = getattr(usage_metadata_obj, 'prompt_token_count', None)
        self._last_completion_tokens = getattr(usage_metadata_obj, 'candidates_token_count', None)

        if self._last_completion_tokens is None:  # Fallback for completion tokens
            if hasattr(usage_metadata_obj, 'total_token_count') and self._last_prompt_tokens is not None:
                self._last_completion_tokens = usage_metadata_obj.total_token_count - self._last_prompt_tokens

        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            logger.info(
                f"  Gemini Token Usage ({self._model_name}): Prompt={self._last_prompt_tokens}, Completion={self._last_completion_tokens}")
        else:
            logger.warning(
                f"  Gemini token usage data incomplete for {self._model_name}. Prompt: {self._last_prompt_tokens}, Completion: {self._last_completion_tokens}, Raw Usage Obj: {usage_metadata_obj}")

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:
        # ... (remains the same as the last correct version) ...
        gemini_history = []
        for msg in history:
            role = 'user' if msg.role == USER_ROLE else ('model' if msg.role == MODEL_ROLE else None)
            if not role:
                logger.debug(f"GeminiAdapter: Skipping message with unhandled role: {msg.role}")
                continue

            text_content = msg.text
            api_parts = []

            if text_content and text_content.strip():
                api_parts.append({'text': text_content})

            if hasattr(msg, 'has_images') and msg.has_images and hasattr(msg, 'image_parts') and msg.image_parts:
                for img_part_data_dict in msg.image_parts:
                    if isinstance(img_part_data_dict, dict) and \
                            img_part_data_dict.get("type") == "image" and \
                            img_part_data_dict.get("mime_type") and \
                            img_part_data_dict.get("data"):
                        try:
                            if API_LIBRARY_AVAILABLE and hasattr(genai, 'types') and hasattr(genai.types, 'Blob'):
                                img_blob = genai.types.Blob(
                                    mime_type=img_part_data_dict["mime_type"],
                                    data=base64.b64decode(img_part_data_dict["data"])
                                )
                                api_parts.append({'inline_data': img_blob})
                            else:
                                logger.warning("Gemini types for Blob not available, cannot format image.")
                        except Exception as e_img_format:
                            logger.warning(f"Could not format image for Gemini: {e_img_format}")

            if api_parts:
                gemini_history.append({"role": role, "parts": api_parts})
            else:
                logger.debug(f"GeminiAdapter: Skipping message for role {role} due to no valid parts (text/image).")

        return gemini_history

    def get_available_models(self) -> List[str]:
        """Get available models with timeout protection"""
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Gemini API library (google-generativeai) not installed."
            return []

        fetched_models_ids: List[str] = []
        try:
            if not self._is_configured:
                logger.warning("GeminiAdapter.get_available_models called before adapter is configured.")

            # NEW: Use threading for timeout protection
            import threading
            import queue

            result_queue = queue.Queue()
            exception_queue = queue.Queue()

            def fetch_models():
                try:
                    models = []
                    for model_info in genai.list_models():
                        if (hasattr(model_info, 'supported_generation_methods') and
                                'generateContent' in model_info.supported_generation_methods and
                                hasattr(model_info, 'name') and "gemini" in model_info.name.lower()):
                            models.append(model_info.name)
                    result_queue.put(models)
                except Exception as e:
                    exception_queue.put(e)

            fetch_thread = threading.Thread(target=fetch_models)
            fetch_thread.daemon = True
            fetch_thread.start()
            fetch_thread.join(timeout=5.0)  # 5 second timeout for Gemini

            if fetch_thread.is_alive():
                logger.warning("Gemini model fetch timed out")
                fetched_models_ids = []
            elif not exception_queue.empty():
                raise exception_queue.get()
            elif not result_queue.empty():
                fetched_models_ids = result_queue.get()

        except PermissionDenied as pde:
            self._last_error = f"Gemini API Permission Denied: {pde}. Check API key."
            logger.error(self._last_error)
            return []
        except Exception as e:
            self._last_error = f"Error fetching Gemini models: {type(e).__name__} - {e}"
            logger.warning(self._last_error)

        # Always provide defaults even if fetch failed
        default_candidates = [
            "models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-latest",
            "models/gemini-pro", "models/gemini-1.0-pro",
        ]
        combined_list = list(set(fetched_models_ids + default_candidates))

        # Keep your existing sorting logic
        def sort_key_gemini(model_name: str):
            name_lower = model_name.lower()
            is_latest = "latest" in name_lower
            is_1_5_pro = "1.5-pro" in name_lower
            is_1_5_flash = "1.5-flash" in name_lower
            is_pro_general = "gemini-pro" in name_lower and not is_1_5_pro and not "1.0-pro" in name_lower
            is_1_0_pro = "1.0-pro" in name_lower
            return (
                0 if is_latest else 1,
                0 if is_1_5_pro else 1 if is_pro_general else 2 if is_1_5_flash else 3 if is_1_0_pro else 4,
                name_lower
            )

        final_model_list = sorted(combined_list, key=sort_key_gemini)

        if self._model_name and self._is_configured and self._model_name not in final_model_list:
            final_model_list.insert(0, self._model_name)
            final_model_list.sort(key=sort_key_gemini)

        return final_model_list
    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None