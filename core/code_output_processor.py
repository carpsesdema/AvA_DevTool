# core/code_output_processor.py - Optimized for speed and accuracy
import logging
import re
import ast
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CodeExtractionStrategy(Enum):
    FENCED_BLOCK = auto()
    MARKED_SECTION = auto()
    LARGEST_BLOCK = auto()
    FULL_RESPONSE = auto()


class CodeQualityLevel(Enum):
    EXCELLENT = auto()  # Perfect syntax, good structure
    GOOD = auto()  # Valid syntax, decent structure
    ACCEPTABLE = auto()  # Valid syntax, basic structure
    POOR = auto()  # Syntax errors but recoverable
    UNUSABLE = auto()  # Too many issues


class CodeOutputProcessor:
    """
    Optimized processor for cleaning and extracting code from LLM responses.
    Focus on speed and accuracy for common patterns.
    """

    def __init__(self):
        self.logger = logger.getChild('CodeProcessor')

        # Pre-compiled regex patterns for performance
        self._fenced_patterns = [
            re.compile(r"```(?:python|py)?\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),
            re.compile(r"```\s*\n(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE),
            re.compile(r"~~~(?:python|py)?\s*\n(.*?)\n?\s*~~~", re.DOTALL | re.IGNORECASE),
        ]

        # Fast cleanup patterns - most common LLM artifacts
        self._cleanup_patterns = [
            re.compile(r"^Here'?s?\s+(?:the\s+)?(?:complete\s+)?(?:python\s+)?code.*?:\s*\n",
                       re.IGNORECASE | re.MULTILINE),
            re.compile(
                r"^(?:The\s+)?(?:complete\s+)?(?:Python\s+)?(?:code\s+)?(?:implementation\s+)?(?:for\s+.*?\s+)?(?:is|would\s+be).*?:\s*\n",
                re.IGNORECASE | re.MULTILINE),
            re.compile(r"^I'?(?:ll|ve)\s+(?:create|write|implement).*?:\s*\n", re.IGNORECASE | re.MULTILINE),
            re.compile(r"\n\s*(?:This|That)\s+(?:code|implementation|solution).*?$", re.IGNORECASE | re.MULTILINE),
            re.compile(r"\n\s*(?:Hope|Let\s+me\s+know).*?$", re.IGNORECASE | re.MULTILINE),
        ]

        # Python keyword detection for fast validation
        self._python_keywords = {
            'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'with', 'as', 'in', 'is', 'and',
            'or', 'not', 'lambda', 'yield', 'async', 'await', 'pass', 'break', 'continue'
        }

        # Cache for repeated validations
        self._validation_cache = {}

    def process_llm_response(self, raw_response: str, filename: str,
                             expected_language: str = "python") -> Tuple[Optional[str], CodeQualityLevel, List[str]]:
        """
        Optimized main entry point for processing LLM code responses.
        Fast path for common patterns, fallback for edge cases.
        """
        if not raw_response or not raw_response.strip():
            return None, CodeQualityLevel.UNUSABLE, ["Empty response"]

        processing_notes = []

        # Fast path: Direct fenced block extraction (90% of cases)
        extracted_code = self._fast_fenced_extraction(raw_response)
        if extracted_code:
            quality = self._assess_code_quality(extracted_code, filename)
            processing_notes.append("Fast fenced block extraction")
            return extracted_code, quality, processing_notes

        # Fallback: Clean and try again
        cleaned_response = self._fast_cleanup(raw_response)
        if cleaned_response != raw_response:
            processing_notes.append("Applied cleanup")

            extracted_code = self._fast_fenced_extraction(cleaned_response)
            if extracted_code:
                quality = self._assess_code_quality(extracted_code, filename)
                processing_notes.append("Extracted after cleanup")
                return extracted_code, quality, processing_notes

        # Last resort: Comprehensive extraction
        return self._comprehensive_extraction(cleaned_response, filename, processing_notes)

    def _fast_fenced_extraction(self, text: str) -> Optional[str]:
        """Optimized fenced block extraction using pre-compiled patterns."""
        largest_block = ""

        for pattern in self._fenced_patterns:
            matches = pattern.findall(text)
            for match in matches:
                content = match.strip() if isinstance(match, str) else match[0].strip()
                if len(content) > len(largest_block) and self._is_valid_code_fast(content):
                    largest_block = content

        return largest_block if largest_block else None

    def _fast_cleanup(self, response: str) -> str:
        """Fast cleanup using pre-compiled patterns."""
        cleaned = response
        for pattern in self._cleanup_patterns:
            cleaned = pattern.sub('', cleaned)
        return cleaned.strip()

    def _is_valid_code_fast(self, code: str) -> bool:
        """Fast validation using cached results and simple heuristics."""
        if len(code) < 10:
            return False

        # Check cache first
        code_hash = hash(code[:100])  # Hash first 100 chars for cache key
        if code_hash in self._validation_cache:
            return self._validation_cache[code_hash]

        # Fast heuristics
        has_keywords = any(f" {kw} " in code or code.startswith(f"{kw} ")
                           for kw in self._python_keywords)
        has_symbols = any(sym in code for sym in ['=', '(', ')', ':', ','])
        has_structure = '\n' in code and any(line.strip().startswith(('def ', 'class ', 'import '))
                                             for line in code.split('\n')[:10])

        is_valid = (has_keywords or has_structure) and has_symbols

        # Cache result
        if len(self._validation_cache) < 100:  # Prevent unbounded growth
            self._validation_cache[code_hash] = is_valid

        return is_valid

    def _comprehensive_extraction(self, text: str, filename: str, notes: List[str]) -> Tuple[
        Optional[str], CodeQualityLevel, List[str]]:
        """Comprehensive extraction for edge cases."""
        # Try marked sections
        marked_patterns = [
            r"(?:Here'?s?\s+(?:the\s+)?(?:complete\s+)?code.*?:\s*\n)(.*?)(?=\n\n|\Z)",
            r"(?:Implementation:\s*\n)(.*?)(?=\n\n|\Z)",
            r"(?:Solution:\s*\n)(.*?)(?=\n\n|\Z)"
        ]

        for pattern in marked_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if self._is_valid_code_fast(extracted):
                    quality = self._assess_code_quality(extracted, filename)
                    notes.append("Marked section extraction")
                    return extracted, quality, notes

        # Try largest code block
        extracted = self._extract_largest_code_block(text)
        if extracted:
            quality = self._assess_code_quality(extracted, filename)
            notes.append("Largest block extraction")
            return extracted, quality, notes

        # Check if entire response is code
        if self._looks_like_pure_code(text):
            quality = self._assess_code_quality(text, filename)
            notes.append("Full response as code")
            return text.strip(), quality, notes

        return None, CodeQualityLevel.UNUSABLE, notes + ["No valid code found"]

    def _extract_largest_code_block(self, text: str) -> Optional[str]:
        """Extract largest code-like block with improved heuristics."""
        lines = text.split('\n')
        current_block = []
        largest_block = ""

        for line in lines:
            stripped = line.strip()

            # Code indicators
            is_code_line = (
                    stripped.startswith(('#', 'def ', 'class ', 'import ', 'from ', '@')) or
                    any(kw in stripped for kw in ['=', 'return', 'if ', 'for ', 'while ']) or
                    line.startswith(('    ', '\t')) or  # Indented
                    stripped.endswith((':', ';', '{', '}'))
            )

            if is_code_line or (not stripped and current_block):  # Include blank lines in code blocks
                current_block.append(line)
            else:
                if len(current_block) > 3:  # Minimum block size
                    block_content = '\n'.join(current_block).strip()
                    if len(block_content) > len(largest_block):
                        largest_block = block_content
                current_block = []

        # Check final block
        if len(current_block) > 3:
            block_content = '\n'.join(current_block).strip()
            if len(block_content) > len(largest_block):
                largest_block = block_content

        return largest_block if largest_block else None

    def _looks_like_pure_code(self, text: str) -> bool:
        """Fast check if text looks like pure code."""
        lines = [line for line in text.strip().split('\n') if line.strip()]
        if len(lines) < 2:
            return False

        code_indicators = 0
        for line in lines[:10]:  # Check first 10 lines for speed
            stripped = line.strip()
            if (stripped.startswith(('#', 'def ', 'class ', 'import ')) or
                    any(kw in stripped for kw in self._python_keywords) or
                    line.startswith(('    ', '\t'))):
                code_indicators += 1

        return code_indicators / min(len(lines), 10) > 0.6

    def _assess_code_quality(self, code: str, filename: str) -> CodeQualityLevel:
        """Optimized code quality assessment."""
        if not code.strip():
            return CodeQualityLevel.UNUSABLE

        try:
            # Primary syntax check
            ast.parse(code)

            # Quick quality indicators
            has_docstrings = '"""' in code or "'''" in code
            has_type_hints = '->' in code and ':' in code
            has_imports = any(line.strip().startswith(('import ', 'from '))
                              for line in code.split('\n')[:5])  # Check first 5 lines only
            has_definitions = any(line.strip().startswith(('def ', 'class '))
                                  for line in code.split('\n')[:10])  # Check first 10 lines only

            # Score calculation
            score = sum([has_docstrings, has_type_hints, has_imports and has_definitions, has_definitions * 0.5])

            if score >= 2.5:
                return CodeQualityLevel.EXCELLENT
            elif score >= 1.5:
                return CodeQualityLevel.GOOD
            else:
                return CodeQualityLevel.ACCEPTABLE

        except SyntaxError as e:
            error_str = str(e).lower()
            # Quick recovery check
            if ('unexpected eof' in error_str or
                    ('invalid syntax' in error_str and len(code.split('\n')) > 5)):
                return CodeQualityLevel.POOR
            return CodeQualityLevel.UNUSABLE
        except Exception:
            return CodeQualityLevel.POOR

    def clean_and_format_code(self, code: str) -> str:
        """Fast cleanup and formatting."""
        if not code:
            return ""

        lines = code.strip().split('\n')

        # Remove empty lines at start and end only
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        return '\n'.join(lines)

    def suggest_fixes_for_poor_code(self, raw_response_text: str, filename: str) -> List[str]:
        """Quick suggestions for common issues."""
        suggestions = []

        if "```" not in raw_response_text:
            suggestions.append("LLM response missing code fences (```)")

        if len(raw_response_text.strip()) < 50:
            suggestions.append("Response too short to contain meaningful code")

        try:
            ast.parse(raw_response_text.strip())
            suggestions.append("Raw response might be valid Python - check extraction logic")
        except SyntaxError:
            suggestions.append("LLM output contains syntax errors")

        if not suggestions:
            suggestions.append("General extraction failure - review LLM output format")

        return suggestions