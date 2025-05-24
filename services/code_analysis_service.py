# services/code_analysis_service.py
import ast
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CodeAnalysisService:
    """
    Service for analyzing code structure to extract functions, classes, and other entities.
    """

    def __init__(self):
        """Initialize the code analysis service."""
        logger.info("CodeAnalysisService initialized")

    def parse_python_structures(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse Python code to extract structural information about functions, classes, etc.

        Args:
            content: The Python source code content
            file_path: Path to the file (for error reporting)

        Returns:
            List of dictionaries containing structure information
        """
        structures = []

        if not content or not content.strip():
            return structures

        try:
            # Parse the Python code into an AST
            tree = ast.parse(content)

            # Extract structures
            structures.extend(self._extract_structures_from_ast(tree, content))

            logger.debug(f"Extracted {len(structures)} code structures from {file_path}")

        except SyntaxError as e:
            logger.warning(f"Syntax error parsing Python file {file_path}: {e}")
            # Try to extract some basic info even with syntax errors
            structures.extend(self._extract_structures_fallback(content))
        except Exception as e:
            logger.error(f"Error parsing Python structures in {file_path}: {e}")
            structures.extend(self._extract_structures_fallback(content))

        return structures

    def _extract_structures_from_ast(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract structures using AST parsing."""
        structures = []
        lines = content.splitlines()

        for node in ast.walk(tree):
            structure_info = None

            if isinstance(node, ast.FunctionDef):
                structure_info = {
                    'type': 'function',
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno,
                    'args': [arg.arg for arg in node.args.args] if hasattr(node.args, 'args') else [],
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef)
                }

            elif isinstance(node, ast.AsyncFunctionDef):
                structure_info = {
                    'type': 'function',
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno,
                    'args': [arg.arg for arg in node.args.args] if hasattr(node.args, 'args') else [],
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'is_async': True
                }

            elif isinstance(node, ast.ClassDef):
                structure_info = {
                    'type': 'class',
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno,
                    'bases': [self._get_name_from_node(base) for base in node.bases],
                    'decorators': [self._get_decorator_name(dec) for dec in node.decorator_list],
                    'methods': []
                }

                # Extract methods from the class
                for class_node in node.body:
                    if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_info = {
                            'name': class_node.name,
                            'start_line': class_node.lineno,
                            'end_line': class_node.end_lineno or class_node.lineno,
                            'is_async': isinstance(class_node, ast.AsyncFunctionDef)
                        }
                        structure_info['methods'].append(method_info)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    structure_info = {
                        'type': 'import',
                        'name': alias.name,
                        'alias': alias.asname,
                        'start_line': node.lineno,
                        'end_line': node.end_lineno or node.lineno
                    }
                    structures.append(structure_info)
                continue  # Don't add to structures again below

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ''
                for alias in node.names:
                    structure_info = {
                        'type': 'import_from',
                        'module': module_name,
                        'name': alias.name,
                        'alias': alias.asname,
                        'start_line': node.lineno,
                        'end_line': node.end_lineno or node.lineno
                    }
                    structures.append(structure_info)
                continue  # Don't add to structures again below

            if structure_info:
                structures.append(structure_info)

        return structures

    def _extract_structures_fallback(self, content: str) -> List[Dict[str, Any]]:
        """Fallback method to extract basic structures using regex when AST fails."""
        import re
        structures = []
        lines = content.splitlines()

        # Simple regex patterns for basic structure detection
        function_pattern = re.compile(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(')
        class_pattern = re.compile(r'^(\s*)class\s+(\w+)(\s*\([^)]*\))?:')
        import_pattern = re.compile(r'^(\s*)(from\s+[\w.]+\s+)?import\s+(.+)')

        for line_num, line in enumerate(lines, 1):
            # Check for function definitions
            func_match = function_pattern.match(line)
            if func_match:
                indent, is_async, func_name = func_match.groups()
                structures.append({
                    'type': 'function',
                    'name': func_name,
                    'start_line': line_num,
                    'end_line': line_num,  # We can't determine end line easily
                    'is_async': bool(is_async),
                    'fallback_parsing': True
                })

            # Check for class definitions
            class_match = class_pattern.match(line)
            if class_match:
                indent, class_name, bases = class_match.groups()
                structures.append({
                    'type': 'class',
                    'name': class_name,
                    'start_line': line_num,
                    'end_line': line_num,  # We can't determine end line easily
                    'fallback_parsing': True
                })

            # Check for imports
            import_match = import_pattern.match(line)
            if import_match:
                indent, from_part, import_part = import_match.groups()
                import_type = 'import_from' if from_part else 'import'

                # Simple parsing of import names
                import_names = [name.strip().split(' as ')[0] for name in import_part.split(',')]
                for name in import_names:
                    if name.strip():
                        structures.append({
                            'type': import_type,
                            'name': name.strip(),
                            'start_line': line_num,
                            'end_line': line_num,
                            'fallback_parsing': True
                        })

        logger.debug(f"Fallback parsing extracted {len(structures)} structures")
        return structures

    def _get_decorator_name(self, decorator_node: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator_node, ast.Name):
            return decorator_node.id
        elif isinstance(decorator_node, ast.Attribute):
            return self._get_name_from_node(decorator_node)
        elif isinstance(decorator_node, ast.Call):
            if isinstance(decorator_node.func, ast.Name):
                return decorator_node.func.id
            elif isinstance(decorator_node.func, ast.Attribute):
                return self._get_name_from_node(decorator_node.func)
        return str(decorator_node)

    def _get_name_from_node(self, node: ast.AST) -> str:
        """Extract name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_name_from_node(node.value)
            return f"{value_name}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)