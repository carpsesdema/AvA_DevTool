# core/user_input_handler.py
import logging
import re
from enum import Enum, auto
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class UserInputIntent(Enum):
    NORMAL_CHAT = auto()
    PLAN_THEN_CODE_REQUEST = auto()
    FILE_CREATION_REQUEST = auto()  # NEW
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
        logger.info("UserInputHandler initialized with enhanced detection logic.")

        # EXPANDED: More comprehensive keywords for plan-then-code
        self.plan_then_code_keywords = [
            # Original keywords
            "generate code for", "bootstrap", "create a script", "develop a module",
            "implement a class", "build an application", "create an app",
            "develop a system", "make a project", "build a tool",
            "create multiple files", "scaffold", "generate project structure",
            "build a complete", "create a full", "develop a complete",
            "make an entire", "build from scratch", "create from scratch",
            "develop an application", "build a web app", "create a web application",
            "make a desktop app", "build a cli tool", "create a command line",
            "develop a library", "build a package", "create a framework",
            "implement a solution", "build a system", "architect a solution",

            # NEW: More natural language patterns that people actually use
            "write code for", "code up", "program a", "write a program",
            "build me a", "create me a", "make me a", "develop me a",
            "i need a program", "i need an app", "i need code",
            "write a python", "create a python", "make a python",
            "write some code", "code me", "program me", "script me",
            "can you code", "can you write", "can you program",
            "please code", "please write", "please program",
            "help me code", "help me write", "help me program",
            "i want to create", "i want to build", "i want to make",
            "write code that", "create code that", "build code that",

            # Function/class specific triggers
            "write a function", "create a function", "make a function",
            "write a class", "create a class", "make a class",
            "implement a function", "implement a class",
            "code a function", "code a class", "program a function",

            # Project/app triggers
            "build an app", "make an app", "create an application",
            "write an application", "develop an app", "code an app"
        ]

        # Patterns for direct file creation requests
        self.file_creation_patterns = [
            # Direct file creation patterns
            r"create (?:a )?file called ['\"]?([a-zA-Z0-9_\-./]+\.(?:py|js|html|css|txt|md|json|yaml|yml))['\"]?",
            r"create ['\"]?([a-zA-Z0-9_\-./]+\.(?:py|js|html|css|txt|md|json|yaml|yml))['\"]?",
            r"make (?:a )?file ['\"]?([a-zA-Z0-9_\-./]+\.(?:py|js|html|css|txt|md|json|yaml|yml))['\"]?",
            r"write (?:a )?file called ['\"]?([a-zA-Z0-9_\-./]+\.(?:py|js|html|css|txt|md|json|yaml|yml))['\"]?",
            r"generate ['\"]?([a-zA-Z0-9_\-./]+\.(?:py|js|html|css|txt|md|json|yaml|yml))['\"]?",
            r"save (?:this )?as ['\"]?([a-zA-Z0-9_\-./]+\.(?:py|js|html|css|txt|md|json|yaml|yml))['\"]?",

            # More natural language patterns
            r"write a (\w+\.py) file",
            r"create a (\w+\.py) script",
            r"make me a (\w+\.py)",
            r"can you create (\w+\.py)",
            r"please create (\w+\.py)",
            r"i need a file called (\w+\.py)",
            r"i want to create (\w+\.py)",
        ]

        # Keywords that indicate single file creation (not multi-file projects)
        self.single_file_keywords = [
            "create a file", "make a file", "write a file", "generate a file",
            "create this file", "save as", "write this as", "make this into",
            "single file", "one file", "just a file", "only a file",
            "simple file", "quick file", "small file"
        ]

    def _detect_file_creation_intent(self, user_text: str) -> Optional[str]:
        """Detect if user wants to create a specific file and return the filename"""
        for pattern in self.file_creation_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                filename = match.group(1)
                logger.info(f"UserInputHandler: Detected file creation intent for: {filename}")
                return filename
        return None

    def _is_plan_then_code_request(self, user_text: str) -> bool:
        """ENHANCED: Determine if this is a plan-then-code request"""
        lower_text = user_text.lower()

        # CRITICAL FIX: Check for project complexity indicators FIRST
        # These override single file patterns and are the key to fixing your issue
        complexity_indicators = [
            # Tool/utility names that suggest complexity
            "file organizer", "organizer", "manager", "system", "application",
            "tool", "utility", "program", "solution", "framework", "analyzer",
            "processor", "generator", "converter", "parser", "scraper", "crawler",
            "scheduler", "monitor", "tracker", "validator", "optimizer", "calculator",
            "compiler", "interpreter", "emulator", "simulator", "installer",

            # Technical complexity
            "with database", "with api", "with frontend", "with backend",
            "with gui", "with interface", "with tests", "with documentation",
            "full stack", "end to end", "complete system", "robust", "comprehensive",
            "production ready", "enterprise", "scalable", "distributed",

            # Project scope indicators
            "multi-file", "multiple files", "project structure", "architecture",
            "full project", "complete solution", "entire system", "whole application"
        ]

        for indicator in complexity_indicators:
            if indicator in lower_text:
                logger.info(f"Found complexity indicator '{indicator}' - routing to plan-then-code")
                return True

        # If it looks like a single file request, check if it's actually complex
        if self._looks_like_single_file_initially(user_text):
            # Check if the "single file" request actually describes something complex
            complex_descriptors = [
                "organizer", "manager", "analyzer", "processor", "generator",
                "converter", "parser", "scraper", "crawler", "scheduler",
                "monitor", "tracker", "validator", "optimizer", "calculator",
                "compiler", "interpreter", "dashboard", "interface", "viewer"
            ]

            for descriptor in complex_descriptors:
                if descriptor in lower_text:
                    logger.info(
                        f"Single file request with complex descriptor '{descriptor}' - upgrading to plan-then-code")
                    return True

            # If it's truly simple, return False (will be handled as single file)
            return False

        # Check for explicit plan-then-code keywords
        for keyword in self.plan_then_code_keywords:
            if keyword in lower_text:
                # Additional check: if it's a simple single function/class request, don't trigger plan-then-code
                simple_patterns = [
                    "write a simple function", "create a quick function", "make a small function",
                    "write a basic class", "create a simple class", "make a basic class"
                ]

                # If it matches a keyword but is asking for just one simple thing, check context
                if any(simple in lower_text for simple in simple_patterns):
                    # If they mention multiple files or complex structure, still do plan-then-code
                    override_indicators = ["multiple", "several", "project", "application", "system"]
                    if not any(indicator in lower_text for indicator in override_indicators):
                        return False

                return True

        # Check for indicators of multi-file projects
        multi_file_indicators = [
            "multiple files", "several files", "project structure", "application",
            "system", "framework", "architecture", "scaffold", "bootstrap",
            "full project", "complete solution", "entire", "whole project",
            "web app", "desktop app", "cli tool", "command line tool",
            "microservice", "api server", "web service", "backend service"
        ]

        for indicator in multi_file_indicators:
            if indicator in lower_text:
                return True

        return False

    def _looks_like_single_file_initially(self, user_text: str) -> bool:
        """Helper to check if something initially looks like single file (before complexity check)"""
        lower_text = user_text.lower()

        single_file_phrases = [
            "create a script", "write a script", "make a script",
            "create a file", "write a file", "make a file",
            "create a function", "write a function", "make a function",
            "create a class", "write a class", "make a class"
        ]

        return any(phrase in lower_text for phrase in single_file_phrases)

    def _is_single_file_request(self, user_text: str) -> bool:
        """ENHANCED: Determine if this is a single file creation request"""
        lower_text = user_text.lower()

        # CRITICAL: Don't treat complex requests as single file
        # Check for complexity first - this prevents the bug you experienced
        complex_indicators = [
            "organizer", "manager", "system", "application", "analyzer",
            "processor", "generator", "converter", "parser", "scraper",
            "comprehensive", "complete", "full", "robust", "advanced",
            "enterprise", "production", "scalable", "framework", "library",
            "dashboard", "interface", "viewer", "monitor", "tracker"
        ]

        for indicator in complex_indicators:
            if indicator in lower_text:
                # This is too complex for single file
                logger.info(f"Request contains complex indicator '{indicator}' - not treating as single file")
                return False

        # Check for explicit single file keywords
        for keyword in self.single_file_keywords:
            if keyword in lower_text:
                logger.info(f"Found single file keyword '{keyword}'")
                return True

        # If a specific filename is mentioned, it's likely a single file request
        if self._detect_file_creation_intent(user_text):
            return True

        # Check for simple code requests that should be single files
        simple_code_patterns = [
            "write a simple function", "create a quick function", "make a small function",
            "write a basic function", "create a basic function", "make a basic function",
            "simple function", "quick function", "small function", "basic function",
            "write a simple class", "create a quick class", "make a small class",
            "write a basic class", "create a basic class", "make a basic class",
            "simple class", "quick class", "small class", "basic class",
            "just a function", "only a function", "one function",
            "just a class", "only a class", "one class"
        ]

        for pattern in simple_code_patterns:
            if pattern in lower_text:
                # Make sure it doesn't have complexity indicators
                complexity_words = ["with database", "with api", "multiple", "complex", "system", "application"]
                if not any(complex_word in lower_text for complex_word in complexity_words):
                    logger.info(f"Found simple code pattern '{pattern}' with no complexity")
                    return True

        # ENHANCED: Simple script patterns - but only if truly simple
        simple_script_patterns = ["write a simple script", "create a quick script", "make a small script"]
        for pattern in simple_script_patterns:
            if pattern in lower_text:
                # Only treat as single file if it's truly simple
                if not any(indicator in lower_text for indicator in complex_indicators):
                    logger.info(f"Found simple script pattern '{pattern}' with no complexity")
                    return True

        return False

    def process_input(self, user_text: str, image_data: Optional[List[Dict[str, Any]]] = None) -> ProcessedInput:
        logger.info(f"UserInputHandler processing: '{user_text[:100]}...'")

        # FIXED PRIORITY ORDER: This is the key fix for your issue

        # 1. First, check for explicit filename creation
        detected_filename = self._detect_file_creation_intent(user_text)

        if detected_filename:
            # Explicit filename mentioned (e.g., "create file_organizer.py")
            logger.info(f"✅ ROUTE: FILE_CREATION_REQUEST for specific file: {detected_filename}")
            return ProcessedInput(
                intent=UserInputIntent.FILE_CREATION_REQUEST,
                data={
                    "user_text": user_text,
                    "image_data": image_data or [],
                    "filename": detected_filename
                },
                original_query=user_text
            )

        # 2. PRIORITY: Check for complex projects FIRST (this fixes your bug)
        elif self._is_plan_then_code_request(user_text):
            logger.info(f"✅ ROUTE: PLAN_THEN_CODE_REQUEST - '{user_text[:50]}...'")
            return ProcessedInput(
                intent=UserInputIntent.PLAN_THEN_CODE_REQUEST,
                data={"user_text": user_text, "image_data": image_data or []},
                original_query=user_text
            )

        # 3. Simple single file request (only if not complex)
        elif self._is_single_file_request(user_text):
            logger.info(f"✅ ROUTE: FILE_CREATION_REQUEST (single file, no filename) - '{user_text[:50]}...'")
            return ProcessedInput(
                intent=UserInputIntent.FILE_CREATION_REQUEST,
                data={
                    "user_text": user_text,
                    "image_data": image_data or [],
                    "filename": None
                },
                original_query=user_text
            )

        # 4. Default to normal chat
        logger.info(f"✅ ROUTE: NORMAL_CHAT - '{user_text[:50]}...'")
        return ProcessedInput(
            intent=UserInputIntent.NORMAL_CHAT,
            data={"user_text": user_text, "image_data": image_data or []},
            original_query=user_text
        )

    def get_intent_description(self, intent: UserInputIntent) -> str:
        """Get a human-readable description of the intent"""
        descriptions = {
            UserInputIntent.NORMAL_CHAT: "Normal conversation with the AI",
            UserInputIntent.PLAN_THEN_CODE_REQUEST: "Multi-file project generation with planning",
            UserInputIntent.FILE_CREATION_REQUEST: "Single file creation request",
            UserInputIntent.UNKNOWN: "Unknown or unrecognized request type"
        }
        return descriptions.get(intent, "Unknown intent type")