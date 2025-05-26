import asyncio
import logging
import os
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
    from google.generativeai.types.generation_types import BlockedPromptException, StopCandidateException
    from google.api_core.exceptions import GoogleAPIError, ClientError, PermissionDenied, ResourceExhausted, \
        InvalidArgument

    API_LIBRARY_AVAILABLE = True
except ImportError:
    genai = None
    HarmCategory = type("HarmCategory", (object,), {})
    HarmBlockThreshold = type("HarmBlockThreshold", (object,), {})
    GenerationConfig = type("GenerationConfig", (object,), {})
    BlockedPromptException = type("BlockedPromptException", (Exception,), {})
    StopCandidateException = type("StopCandidateException", (Exception,), {})
    GoogleAPIError = type("GoogleAPIError", (Exception,), {})
    ClientError = type("ClientError", (GoogleAPIError,), {})
    PermissionDenied = type("PermissionDenied", (ClientError,), {})
    ResourceExhausted = type("ResourceExhausted", (ClientError,), {})
    InvalidArgument = type("InvalidArgument", (ClientError,), {})
    API_LIBRARY_AVAILABLE = False
    logging.getLogger(__name__).warning("GeminiAdapter: google-generativeai library not found.")

logger = logging.getLogger(__name__)


class GeminiAdapter(BackendInterface):
    def __init__(self):
        self._model: Optional[genai.GenerativeModel] = None  # type: ignore
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
        self._model = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Gemini API library (google-generativeai) not installed."
            return False

        env_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key and not env_api_key:
            self._last_error = "Gemini API key not provided and not found in GOOGLE_API_KEY/GEMINI_API_KEY environment variables."
            logger.error(self._last_error)
            return False

        if not model_name:
            self._last_error = "Model name is required for Gemini configuration."
            return False

        try:
            current_api_key = api_key if api_key and api_key.strip() else env_api_key
            if current_api_key:  # Only configure if a key is actually present
                genai.configure(api_key=current_api_key)  # type: ignore

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,  # type: ignore
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,  # type: ignore
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,  # type: ignore
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,  # type: ignore
            }
            effective_system_instruction = system_prompt.strip() if system_prompt and system_prompt.strip() else None

            self._model = genai.GenerativeModel(  # type: ignore
                model_name=model_name,
                safety_settings=safety_settings,
                system_instruction=effective_system_instruction
            )
            self._model_name = model_name
            self._system_prompt = effective_system_instruction
            self._is_configured = True
            return True
        except ValueError as ve:
            self._last_error = f"Configuration Error (ValueError): {ve}."
        except InvalidArgument as iae:  # type: ignore
            self._last_error = f"API Error (Invalid Argument): {iae}."
        except PermissionDenied as pde:  # type: ignore
            self._last_error = f"API Error (Permission Denied): {pde}."
        except Exception as e:
            self._last_error = f"Unexpected error configuring Gemini: {type(e).__name__} - {e}"

        self._is_configured = False
        return False

    def is_configured(self) -> bool:
        return self._is_configured

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self,
                                  history: List[ChatMessage],  # type: ignore
                                  options: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._model:
            self._last_error = "Adapter not configured or Gemini model object missing."
            raise RuntimeError(self._last_error)

        gemini_history_api_format = self._format_history_for_api(history)
        if not gemini_history_api_format and not self._system_prompt:
            self._last_error = "Cannot send request: No valid messages and no system instruction."
            raise ValueError(self._last_error)

        generation_config_dict = {}
        if options and "temperature" in options and isinstance(options["temperature"], (float, int)):
            generation_config_dict["temperature"] = float(options["temperature"])

        effective_generation_config = GenerationConfig(
            **generation_config_dict) if generation_config_dict else None  # type: ignore

        try:
            def _initiate_stream_call_in_thread():
                return self._model.generate_content(  # type: ignore
                    contents=gemini_history_api_format,  # type: ignore
                    stream=True,
                    generation_config=effective_generation_config
                )

            stream_response = await asyncio.to_thread(_initiate_stream_call_in_thread)

            # FIXED: Process stream chunk by chunk with periodic yielding
            chunk_count = 0
            for chunk in stream_response:
                chunk_count += 1

                if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and \
                        hasattr(chunk.prompt_feedback, 'block_reason') and chunk.prompt_feedback.block_reason:
                    err_msg = f"Content blocked (prompt feedback): {chunk.prompt_feedback.block_reason}."
                    self._last_error = err_msg
                    yield f"[SYSTEM ERROR: {err_msg}]"
                    return

                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason and \
                                candidate.finish_reason.name not in ["STOP", "MAX_TOKENS", "UNSPECIFIED", "NULL"]:
                            err_msg = f"Generation stopped. Reason: {candidate.finish_reason.name}."
                            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                err_msg += f" Details: {candidate.safety_ratings}"
                            self._last_error = err_msg
                            yield f"[SYSTEM ERROR: {err_msg}]"
                            return

                text_parts_from_chunk = []
                if hasattr(chunk, 'parts') and chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text is not None:
                            text_parts_from_chunk.append(part.text)
                elif hasattr(chunk, 'text') and chunk.text is not None:
                    text_parts_from_chunk.append(chunk.text)
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    for cand in chunk.candidates:
                        if hasattr(cand, 'content') and cand.content and \
                                hasattr(cand.content, 'parts') and cand.content.parts:
                            for part in cand.content.parts:
                                if hasattr(part, 'text') and part.text is not None:
                                    text_parts_from_chunk.append(part.text)

                if text_parts_from_chunk:
                    yield "".join(text_parts_from_chunk)

                # CRITICAL: Yield control every few chunks to prevent GUI blocking
                if chunk_count % 3 == 0:
                    await asyncio.sleep(0)

            # Get usage stats from the final response
            if hasattr(stream_response, 'usage_metadata') and stream_response.usage_metadata:
                usage_sdk = stream_response.usage_metadata
                self._last_prompt_tokens = getattr(usage_sdk, 'prompt_token_count', None)
                completion_val = getattr(usage_sdk, 'candidates_token_count', None)
                if completion_val is None and hasattr(usage_sdk, 'result'):
                    res_obj = getattr(usage_sdk, 'result', {})
                    if isinstance(res_obj, dict):
                        completion_val = res_obj.get('candidates_token_count', None)
                if completion_val is None and hasattr(usage_sdk,
                                                      'total_token_count') and self._last_prompt_tokens is not None:
                    self._last_completion_tokens = usage_sdk.total_token_count - self._last_prompt_tokens
                else:
                    self._last_completion_tokens = completion_val

        except BlockedPromptException as bpe:  # type: ignore
            self._last_error = f"API Error: Prompt blocked. {bpe.args}"
            raise RuntimeError(self._last_error) from bpe
        except InvalidArgument as iae:  # type: ignore
            self._last_error = f"API Error (Invalid Argument): {iae}."
            raise RuntimeError(self._last_error) from iae
        except PermissionDenied as pde:  # type: ignore
            self._last_error = f"API Error (Permission Denied): {pde}."
            raise RuntimeError(self._last_error) from pde
        except Exception as e_init_stream:
            self._last_error = f"Error during Gemini stream: {type(e_init_stream).__name__} - {e_init_stream}"
            raise RuntimeError(self._last_error) from e_init_stream

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:  # type: ignore
        gemini_history = []
        for msg in history:  # type: ignore
            role = 'user' if msg.role == USER_ROLE else ('model' if msg.role == MODEL_ROLE else None)
            if not role: continue
            text_content = msg.text
            if not text_content or not text_content.strip(): continue
            api_parts = [{'text': text_content}]
            gemini_history.append({"role": role, "parts": api_parts})
        return gemini_history

    def get_available_models(self) -> List[str]:
        self._last_error = None
        if not API_LIBRARY_AVAILABLE:
            self._last_error = "Gemini API library not installed."
            return []

        fetched_models_ids: List[str] = []
        try:
            if not self._is_configured and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                pass  # Let genai.list_models try and handle auth

            for model_info in genai.list_models():  # type: ignore
                if 'generateContent' in model_info.supported_generation_methods and "gemini" in model_info.name.lower():
                    fetched_models_ids.append(model_info.name)
        except PermissionDenied as pde:  # type: ignore
            self._last_error = f"API Permission Denied listing models: {pde}."
            return []
        except InvalidArgument as iae:  # type: ignore
            self._last_error = f"API Invalid Argument listing models: {iae}."
            return []
        except GoogleAPIError as api_err:  # type: ignore
            self._last_error = f"Google API Error listing models: {type(api_err).__name__} - {api_err}"
            return []
        except Exception as e:
            self._last_error = f"Unexpected error fetching Gemini models: {type(e).__name__} - {e}"
            return []

        default_candidates = [
            "models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-latest", "models/gemini-1.0-pro",
        ]
        final_model_list = sorted(list(set(fetched_models_ids + default_candidates)),
                                  key=lambda x: (0 if "latest" in x.lower() else 1,
                                                 0 if "1.5-pro" in x.lower() else 1 if "gemini-pro" in x.lower() and "1.0" not in x.lower() else 2 if "1.5-flash" in x.lower() else 3 if "1.0-pro" in x.lower() else 4,
                                                 x.lower()))

        if self._model_name and self._is_configured and self._model_name not in final_model_list:
            final_model_list.insert(0, self._model_name)
            final_model_list.sort(key=lambda x: (0 if "latest" in x.lower() else 1,
                                                 0 if "1.5-pro" in x.lower() else 1 if "gemini-pro" in x.lower() and "1.0" not in x.lower() else 2 if "1.5-flash" in x.lower() else 3 if "1.0-pro" in x.lower() else 4,
                                                 x.lower()))

        return final_model_list

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None