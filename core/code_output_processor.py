# core/code_output_processor.py
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
    OLLAMA_SPECIFIC = auto()


class CodeQualityLevel(Enum):
    EXCELLENT = auto()  # Perfect syntax, good structure
    GOOD = auto()  # Valid syntax, decent structure
    ACCEPTABLE = auto()  # Valid syntax, basic structure
    POOR = auto()  # Syntax errors but recoverable
    UNUSABLE = auto()  # Too many issues


class CodeOutputProcessor:
    """
    Dedicated processor for cleaning and extracting code from LLM responses.
    Handles various LLM output formats and streaming response artifacts.
    """

    def __init__(self):
        self.logger = logger.getChild('CodeProcessor')

        # Enhanced patterns for Ollama-specific artifacts
        self.ollama_artifacts = [
            r"```\s*(?:python|py)?\s*\n",  # Opening fences
            r"\n\s*```\s*$",  # Closing fences
            r"^Here'?s?\s+(?:the\s+)?(?:complete\s+)?(?:python\s+)?code.*?:\s*\n",
            r"^(?:The\s+)?(?:complete\s+)?(?:Python\s+)?(?:code\s+)?(?:implementation\s+)?(?:for\s+.*?\s+)?(?:is|would\s+be).*?:\s*\n",
            r"^I'?(?:ll|ve)\s+(?:create|write|implement).*?:\s*\n",
            r"^(?:Below|Here)\s+is\s+(?:the\s+)?(?:complete\s+)?.*?:\s*\n",
            r"\n\s*(?:This|That)\s+(?:code|implementation|solution).*?$",
            r"\n\s*(?:Hope|Let\s+me\s+know)\s+.*?$",
            r"\n\s*(?:Please|Make\s+sure)\s+.*?$",
        ]

        # Compile patterns for performance
        self.compiled_artifacts = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                                   for pattern in self.ollama_artifacts]

        # Code block extraction patterns (ordered by preference)
        self.code_patterns = [
            (CodeExtractionStrategy.FENCED_BLOCK, [
                r"```python\n(.*?)```",
                r"```py\n(.*?)```",
                r"```\n(.*?)```",
                r"~~~python\n(.*?)~~~",
                r"~~~\n(.*?)~~~"
            ]),
            (CodeExtractionStrategy.MARKED_SECTION, [
                r"(?:Here'?s?\s+(?:the\s+)?(?:complete\s+)?code.*?:\s*\n)(.*?)(?=\n\n|\Z)",
                r"(?:The\s+(?:complete\s+)?Python\s+code.*?:\s*\n)(.*?)(?=\n\n|\Z)",
                r"(?:Implementation:\s*\n)(.*?)(?=\n\n|\Z)",
                r"(?:Solution:\s*\n)(.*?)(?=\n\n|\Z)"
            ])
        ]

    def process_llm_response(self, raw_response: str, filename: str,
                             expected_language: str = "python") -> Tuple[Optional[str], CodeQualityLevel, List[str]]:
        """
        Main entry point for processing LLM code responses.

        Returns:
            Tuple of (extracted_code, quality_level, processing_notes)
        """
        if not raw_response or not raw_response.strip():
            return None, CodeQualityLevel.UNUSABLE, ["Empty response"]

        processing_notes = []

        # Step 1: Clean Ollama-specific artifacts
        cleaned_response = self._clean_ollama_artifacts(raw_response)
        if cleaned_response != raw_response:
            processing_notes.append("Removed Ollama streaming artifacts")

        # Step 2: Try extraction strategies in order of preference
        for strategy, patterns in self.code_patterns:
            extracted_code = self._try_extraction_strategy(cleaned_response, patterns, strategy)
            if extracted_code:
                quality = self._assess_code_quality(extracted_code, filename)
                processing_notes.append(f"Extracted using {strategy.name}")
                return extracted_code, quality, processing_notes

        # Step 3: Fallback to largest code-like block
        extracted_code = self._extract_largest_code_block(cleaned_response)
        if extracted_code:
            quality = self._assess_code_quality(extracted_code, filename)
            processing_notes.append("Extracted largest code-like block")
            return extracted_code, quality, processing_notes

        # Step 4: Last resort - use entire response if it looks like code
        if self._looks_like_pure_code(cleaned_response):
            quality = self._assess_code_quality(cleaned_response, filename)
            processing_notes.append("Used entire response as code")
            return cleaned_response.strip(), quality, processing_notes

        return None, CodeQualityLevel.UNUSABLE, processing_notes + ["No valid code found"]

    def _clean_ollama_artifacts(self, response: str) -> str:
        """Remove common Ollama streaming artifacts and explanatory text."""
        cleaned = response

        for pattern in self.compiled_artifacts:
            cleaned = pattern.sub('', cleaned)

        # Remove leading/trailing whitespace and empty lines
        lines = cleaned.split('\n')

        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        return '\n'.join(lines)

    def _try_extraction_strategy(self, text: str, patterns: List[str],
                                 strategy: CodeExtractionStrategy) -> Optional[str]:
        """Try a specific extraction strategy with multiple patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if self._is_valid_code_candidate(extracted):
                    self.logger.debug(f"Strategy {strategy.name} succeeded with pattern: {pattern[:30]}...")
                    return extracted
        return None

    def _extract_largest_code_block(self, text: str) -> Optional[str]:
        """Extract the largest block that looks like code."""
        lines = text.split('\n')
        current_block = []
        largest_block = []

        for line in lines:
            stripped = line.strip()

            # Skip obvious non-code lines
            if self._is_non_code_line(stripped):
                if len(current_block) > len(largest_block):
                    largest_block = current_block[:]
                current_block = []
                continue

            # Collect potential code lines
            if self._is_code_like_line(stripped) or (stripped and len(current_block) > 0):
                current_block.append(line)
            elif stripped:  # Non-empty line that might be code
                current_block.append(line)

        # Check the last block
        if len(current_block) > len(largest_block):
            largest_block = current_block[:]

        if largest_block and len(largest_block) >= 3:  # Minimum viable code block
            return '\n'.join(largest_block).strip()

        return None

    def _is_non_code_line(self, line: str) -> bool:
        """Check if a line is clearly not code."""
        if not line:
            return False

        non_code_indicators = [
            line.startswith(('Here', 'The ', 'This ', 'I ', 'Let ', 'Note:', 'Please')),
            '```' in line,
            line.endswith('?'),
            line.lower().startswith(('explanation:', 'usage:', 'example:')),
            re.match(r'^[A-Z][a-z\s]+[.!]$', line)  # Sentence-like
        ]

        return any(non_code_indicators)

    def _is_code_like_line(self, line: str) -> bool:
        """Check if a line looks like code."""
        if not line:
            return False

        code_indicators = [
            line.startswith(('#', 'def ', 'class ', 'import ', 'from ', '@')),
            any(keyword in line for keyword in
                ['=', 'return', 'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except']),
            line.startswith(('    ', '\t')),  # Indented
        ]

        return any(code_indicators)

    def _is_valid_code_candidate(self, code: str) -> bool:
        """Quick validation that extracted text could be code."""
        if not code or len(code.strip()) < 10:
            return False

        # Must have some Python-like characteristics
        python_indicators = [
            'def ', 'class ', 'import ', 'from ', '=', 'return',
            'if ', 'else:', 'try:', 'except', '#'
        ]

        indicator_count = sum(1 for indicator in python_indicators if indicator in code)
        return indicator_count >= 2

    def _looks_like_pure_code(self, text: str) -> bool:
        """Check if entire text looks like pure code without explanations."""
        lines = text.strip().split('\n')
        if len(lines) < 3:
            return False

        code_lines = sum(1 for line in lines if self._is_code_like_line(line.strip()))
        non_code_lines = sum(1 for line in lines if self._is_non_code_line(line.strip()))

        # Should be mostly code lines
        return code_lines > max(3, len(lines) * 0.7) and non_code_lines < len(lines) * 0.2

    def _assess_code_quality(self, code: str, filename: str) -> CodeQualityLevel:
        """Assess the quality of extracted code."""
        try:
            # Try to parse with AST
            ast.parse(code)

            # Additional quality checks
            lines = code.split('\n')

            # Check for good practices
            has_docstrings = '"""' in code or "'''" in code
            has_type_hints = ':' in code and '->' in code
            has_proper_imports = any(line.strip().startswith(('import ', 'from ')) for line in lines)
            has_functions_or_classes = any(line.strip().startswith(('def ', 'class ')) for line in lines)

            quality_score = 0
            if has_docstrings: quality_score += 1
            if has_type_hints: quality_score += 1
            if has_proper_imports: quality_score += 1
            if has_functions_or_classes: quality_score += 1

            if quality_score >= 3:
                return CodeQualityLevel.EXCELLENT
            elif quality_score >= 2:
                return CodeQualityLevel.GOOD
            else:
                return CodeQualityLevel.ACCEPTABLE

        except SyntaxError as e:
            # Check if it's recoverable
            if self._is_recoverable_syntax_error(code, str(e)):
                return CodeQualityLevel.POOR
            else:
                return CodeQualityLevel.UNUSABLE
        except Exception:
            return CodeQualityLevel.POOR

    def _is_recoverable_syntax_error(self, code: str, error: str) -> bool:
        """Check if syntax error might be recoverable."""
        # Common recoverable issues
        recoverable_indicators = [
            'unexpected EOF' in error.lower(),
            'invalid syntax' in error.lower() and len(code.split('\n')) > 5,
            'indentation' in error.lower()
        ]

        return any(recoverable_indicators)

    def clean_and_format_code(self, code: str) -> str:
        """Final cleanup and formatting of extracted code."""
        if not code:
            return code

        lines = code.split('\n')

        # Remove excessive blank lines
        cleaned_lines = []
        prev_blank = False

        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue  # Skip consecutive blank lines
            cleaned_lines.append(line)
            prev_blank = is_blank

        # Remove trailing blank lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        return '\n'.join(cleaned_lines)

    def suggest_fixes_for_poor_code(self, code: str, filename: str) -> List[str]:
        """Suggest potential fixes for poor quality code."""
        suggestions = []

        try:
            ast.parse(code)
        except SyntaxError as e:
            error_msg = str(e)
            if 'unexpected EOF' in error_msg:
                suggestions.append("Code appears incomplete - may need additional lines")
            elif 'invalid syntax' in error_msg:
                suggestions.append(f"Syntax error at line {e.lineno}: {error_msg}")
            elif 'indentation' in error_msg:
                suggestions.append("Indentation issues detected - check whitespace consistency")

        # Check for common issues
        if not any(line.strip().startswith(('import ', 'from ')) for line in code.split('\n')):
            suggestions.append("Missing import statements")

        if 'def ' not in code and 'class ' not in code:
            suggestions.append("No functions or classes found - might be incomplete")

        return suggestions