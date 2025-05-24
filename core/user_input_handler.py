# core/user_input_handler.py
import logging
from enum import Enum, auto
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class UserInputIntent(Enum):
    NORMAL_CHAT = auto()
    PLAN_THEN_CODE_REQUEST = auto()
    UNKNOWN = auto()


class ProcessedInput:
    def __init__(self, intent: UserInputIntent, data: Optional[Dict[str, Any]] = None, original_query: str = ""):
        self.intent = intent
        self.data = data if data is not None else {}
        self.original_query = original_query

    def __repr__(self):
        return f"ProcessedInput(intent={self.intent.name}, data_keys={list(self.data.keys())}, original_query_preview='{self.original_query[:30]}...')"


class UserInputHandler:
    def __init__(self):
        logger.info("UserInputHandler initialized.")
        self.plan_then_code_keywords = ["generate code for", "bootstrap", "create a script", "develop a module",
                                        "implement a class"]  # Can be expanded

    def process_input(self, user_text: str, image_data: Optional[List[Dict[str, Any]]] = None) -> ProcessedInput:
        logger.debug(f"UserInputHandler processing text: '{user_text[:50]}...'")

        lower_user_text = user_text.lower()

        for keyword in self.plan_then_code_keywords:
            if keyword in lower_user_text:
                logger.info(f"Detected PLAN_THEN_CODE_REQUEST due to keyword: '{keyword}'")
                return ProcessedInput(
                    intent=UserInputIntent.PLAN_THEN_CODE_REQUEST,
                    data={"user_text": user_text, "image_data": image_data or []},
                    original_query=user_text
                )

        logger.info("Defaulting to NORMAL_CHAT intent.")
        return ProcessedInput(
            intent=UserInputIntent.NORMAL_CHAT,
            data={"user_text": user_text, "image_data": image_data or []},
            original_query=user_text
        )