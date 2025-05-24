import asyncio
import logging
import os
from typing import List, Optional, AsyncGenerator, Dict, Any, Tuple

try:
    from backends.backend_interface import BackendInterface
    from core.models import ChatMessage, MODEL_ROLE, USER_ROLE, SYSTEM_ROLE
except ImportError:
    BackendInterface = type("BackendInterface", (object,), {})
    ChatMessage = type("ChatMessage", (object,), {})
    MODEL_ROLE, USER_ROLE, SYSTEM_ROLE = "model", "user", "system"

try:
    import openai
    from openai import APIError, AuthenticationError, RateLimitError, NotFoundError, APIConnectionError, \
        APITimeoutError  # type: ignore

    OPENAI_API_LIBRARY_AVAILABLE = True
except ImportError:
    openai = None
    APIError = type("APIError", (Exception,), {})
    AuthenticationError = type("AuthenticationError", (APIError,), {})
    RateLimitError = type("RateLimitError", (APIError,), {})
    NotFoundError = type("NotFoundError", (APIError,), {})
    APIConnectionError = type("APIConnectionError", (APIError,), {})
    APITimeoutError = type("APITimeoutError", (APIError,), {})
    OPENAI_API_LIBRARY_AVAILABLE = False
    logging.getLogger(__name__).warning("GPTAdapter: 'openai' library not found. Please install it: pip install openai")

logger = logging.getLogger(__name__)
_SENTINEL_GPT = object()


def _blocking_next_or_sentinel_gpt(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL_GPT
    except Exception as e_next:
        raise RuntimeError(f"Error in OpenAI stream iterator: {type(e_next).__name__} - {e_next}") from e_next


class GPTAdapter(BackendInterface):
    def __init__(self):
        self._client: Optional[openai.OpenAI] = None  # type: ignore
        self._model_name: Optional[str] = None
        self._system_prompt: Optional[str] = None
        self._last_error: Optional[str] = None
        self._is_configured: bool = False
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: Optional[int] = None
        logger.info("GPTAdapter initialized.")

    def configure(self, api_key: Optional[str], model_name: str, system_prompt: Optional[str] = None) -> bool:
        self._client = None
        self._is_configured = False
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not OPENAI_API_LIBRARY_AVAILABLE:
            self._last_error = "OpenAI API library ('openai') not installed."
            return False

        effective_api_key = api_key
        if not effective_api_key or not effective_api_key.strip():
            effective_api_key = os.getenv("OPENAI_API_KEY")
            if not effective_api_key or not effective_api_key.strip():
                self._last_error = "OpenAI API key not provided and not in OPENAI_API_KEY environment variable."
                return False

        if not model_name:
            self._last_error = "Model name is required for GPT configuration."
            return False

        try:
            self._client = openai.OpenAI(api_key=effective_api_key)  # type: ignore
            self._model_name = model_name
            self._system_prompt = system_prompt.strip() if isinstance(system_prompt,
                                                                      str) and system_prompt.strip() else None
            self._is_configured = True
            return True
        except AuthenticationError as e:  # type: ignore
            self._last_error = f"OpenAI Authentication Error: {e}. Check API key."
        except APIConnectionError as e:  # type: ignore
            self._last_error = f"OpenAI API Connection Error: {e}."
        except Exception as e:
            self._last_error = f"Unexpected error configuring OpenAI '{model_name}': {type(e).__name__} - {e}"

        self._is_configured = False
        return False

    def is_configured(self) -> bool:
        return self._is_configured

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    async def get_response_stream(self, history: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> \
    AsyncGenerator[str, None]:  # type: ignore
        self._last_error = None
        self._last_prompt_tokens = None
        self._last_completion_tokens = None

        if not self.is_configured() or not self._client:
            self._last_error = "GPTAdapter is not configured or client missing."
            raise RuntimeError(self._last_error)

        messages_for_api = self._format_history_for_api(history)  # type: ignore
        if not messages_for_api and not self._system_prompt:
            self._last_error = "Cannot send request: No valid messages and no system prompt."
            raise ValueError(self._last_error)

        api_params: Dict[str, Any] = {"model": self._model_name, "messages": messages_for_api, "stream": True}
        if options:
            if "temperature" in options and isinstance(options["temperature"], (float, int)):
                api_params["temperature"] = float(options["temperature"])
            if "max_tokens" in options and isinstance(options["max_tokens"], int):
                api_params["max_tokens"] = options["max_tokens"]

        sync_iterator = None
        try:
            def _initiate_openai_stream_call_in_thread():
                return self._client.chat.completions.create(**api_params)  # type: ignore

            sync_iterator = await asyncio.to_thread(_initiate_openai_stream_call_in_thread)

            while True:
                chunk_obj = await asyncio.to_thread(_blocking_next_or_sentinel_gpt, sync_iterator)
                if chunk_obj is _SENTINEL_GPT: break
                if not hasattr(chunk_obj, 'choices') or not chunk_obj.choices: continue  # type: ignore

                delta = chunk_obj.choices[0].delta  # type: ignore
                finish_reason = chunk_obj.choices[0].finish_reason  # type: ignore

                if delta and delta.content: yield delta.content

                if hasattr(chunk_obj, 'usage') and chunk_obj.usage:  # type: ignore
                    self._last_prompt_tokens = chunk_obj.usage.prompt_tokens  # type: ignore
                    self._last_completion_tokens = chunk_obj.usage.completion_tokens  # type: ignore

                if finish_reason: break
        except AuthenticationError as e:  # type: ignore
            self._last_error = f"OpenAI API Authentication Error: {e}"
            raise RuntimeError(self._last_error) from e
        except RateLimitError as e:  # type: ignore
            self._last_error = f"OpenAI API Rate Limit Error: {e}"
            raise RuntimeError(self._last_error) from e
        except APIConnectionError as e:  # type: ignore
            self._last_error = f"OpenAI API Connection Error: {e}"
            raise RuntimeError(self._last_error) from e
        except APITimeoutError as e:  # type: ignore
            self._last_error = f"OpenAI API Timeout Error: {e}"
            raise RuntimeError(self._last_error) from e
        except NotFoundError as e:  # type: ignore
            self._last_error = f"OpenAI API Not Found Error (model '{self._model_name}' invalid?): {e}"
            raise RuntimeError(self._last_error) from e
        except APIError as e:  # type: ignore
            self._last_error = f"OpenAI API Error: {type(e).__name__} - {e}"
            raise RuntimeError(self._last_error) from e
        except RuntimeError as e_rt:
            if not self._last_error: self._last_error = f"Runtime error during OpenAI stream: {e_rt}"
            raise
        except Exception as e_general:
            if not self._last_error: self._last_error = f"Unexpected error in OpenAI stream: {type(e_general).__name__} - {e_general}"
            raise RuntimeError(self._last_error) from e_general

    def _format_history_for_api(self, history: List[ChatMessage]) -> List[Dict[str, Any]]:  # type: ignore
        openai_messages: List[Dict[str, Any]] = []
        if self._system_prompt:
            openai_messages.append({"role": "system", "content": self._system_prompt})

        for msg in history:  # type: ignore
            role_for_api: Optional[str] = None
            if msg.role == USER_ROLE:
                role_for_api = "user"
            elif msg.role == MODEL_ROLE:
                role_for_api = "assistant"
            elif msg.role == SYSTEM_ROLE:
                if not self._system_prompt:
                    role_for_api = "system"
                else:
                    continue
            else:
                continue

            text_content = msg.text  # type: ignore
            message_content_parts_for_api: List[Dict[str, Any]] = []
            if text_content and text_content.strip():
                message_content_parts_for_api.append({"type": "text", "text": text_content})

            if hasattr(msg, 'has_images') and msg.has_images and hasattr(msg,
                                                                         'image_parts') and msg.image_parts:  # type: ignore
                is_vision_model = self._model_name and (
                            "vision" in self._model_name or "gpt-4-turbo" in self._model_name or "gpt-4o" in self._model_name)  # type: ignore
                if is_vision_model:
                    for img_part_dict in msg.image_parts:  # type: ignore
                        if isinstance(img_part_dict, dict) and img_part_dict.get("type") == "image" and \
                                img_part_dict.get("mime_type") and img_part_dict.get("data"):
                            image_url_data = f"data:{img_part_dict['mime_type']};base64,{img_part_dict['data']}"
                            message_content_parts_for_api.append(
                                {"type": "image_url", "image_url": {"url": image_url_data}})

            final_content_value_for_api: Any
            if not message_content_parts_for_api:
                if role_for_api in ["user", "assistant"]:
                    continue
                else:
                    final_content_value_for_api = ""
            elif len(message_content_parts_for_api) == 1 and message_content_parts_for_api[0]["type"] == "text":
                final_content_value_for_api = message_content_parts_for_api[0]["text"]
            else:
                final_content_value_for_api = message_content_parts_for_api

            openai_messages.append({"role": role_for_api, "content": final_content_value_for_api})
        return openai_messages

    def get_available_models(self) -> List[str]:
        self._last_error = None
        if not OPENAI_API_LIBRARY_AVAILABLE:
            self._last_error = "OpenAI API library ('openai') not installed."
            return []
        if not self.is_configured() or not self._client:
            self._last_error = "GPTAdapter not configured (API key likely missing)."
            return []

        fetched_models: List[str] = []
        try:
            model_list_response = self._client.models.list()  # type: ignore
            chat_model_indicators = ("gpt-4", "gpt-3.5-turbo")
            non_chat_keywords = ("embedding", "instruct", "davinci", "curie", "babbage", "ada", "text-", "code-",
                                 "edit-", "audio", "image", "-dalle")

            for model_obj in model_list_response.data:  # type: ignore
                model_id_lower = model_obj.id.lower()
                is_potential_chat_model = any(indicator in model_id_lower for indicator in chat_model_indicators)
                if is_potential_chat_model:
                    is_non_chat_type = any(kw in model_id_lower for kw in non_chat_keywords if
                                           not any(indicator in model_id_lower for indicator in chat_model_indicators))
                    if "instruct" in model_id_lower and "turbo" not in model_id_lower and "gpt-4" not in model_id_lower: is_non_chat_type = True
                    if model_id_lower.startswith(
                        "text-") and "turbo" not in model_id_lower and "gpt-4" not in model_id_lower: is_non_chat_type = True
                    if not is_non_chat_type: fetched_models.append(model_obj.id)
        except AuthenticationError as e:  # type: ignore
            self._last_error = f"OpenAI API Authentication Error listing models: {e}"
        except APIError as e:  # type: ignore
            self._last_error = f"OpenAI API Error listing models: {e}"
        except Exception as e:
            self._last_error = f"Unexpected error fetching OpenAI models: {type(e).__name__} - {e}"

        if self._last_error: return []

        final_model_list = sorted(list(set(fetched_models)), key=lambda x: (
            0 if "gpt-4o" in x else 1 if "gpt-4-turbo" in x else 2 if "gpt-4" in x else 3 if "gpt-3.5-turbo-16k" in x else 4 if "gpt-3.5-turbo" in x else 5,
            x
        ))
        if self._model_name and self._is_configured and self._model_name not in final_model_list:
            final_model_list.insert(0, self._model_name)
            final_model_list.sort(key=lambda x: (
                0 if "gpt-4o" in x else 1 if "gpt-4-turbo" in x else 2 if "gpt-4" in x else 3 if "gpt-3.5-turbo-16k" in x else 4 if "gpt-3.5-turbo" in x else 5,
                x))
        return final_model_list

    def get_last_token_usage(self) -> Optional[Tuple[int, int]]:
        if self._last_prompt_tokens is not None and self._last_completion_tokens is not None:
            return (self._last_prompt_tokens, self._last_completion_tokens)
        return None

