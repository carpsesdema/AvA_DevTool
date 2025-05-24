# core/rag_handler.py
import logging
import os
import re
from typing import List, Optional, Set, Tuple, Dict, Any  # Added Dict, Any

try:
    from services.upload_service import UploadService
    from services.vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID
    from utils import constants
except ImportError as e:
    logging.critical(f"RagHandler: Failed to import services/utils: {e}")
    UploadService = type("UploadService", (object,), {})  # type: ignore
    VectorDBService = type("VectorDBService", (object,), {})  # type: ignore
    GLOBAL_COLLECTION_ID = "global_collection"
    constants = type("constants", (object,),  # type: ignore
                     {"RAG_NUM_RESULTS": 5, "RAG_CHUNK_SIZE": 1000, "RAG_CHUNK_OVERLAP": 150})

logger = logging.getLogger(__name__)


class RagHandler:
    _TECHNICAL_KEYWORDS = {
        'python', 'code', 'error', 'fix', 'implement', 'explain', 'how to',
        'def ', 'class ', 'import ', ' module', ' function', ' method',
        ' attribute', ' bug', ' issue', ' traceback', ' install', ' pip',
        ' library', ' package', ' api', ' request', ' data', ' typeerror',
        ' indexerror', ' keyerror', ' exception', ' syntax', ' logic', ' algorithm',
        ' self.', ' args', ' kwargs', ' return', ' yield', ' async', ' await',
        ' decorator', ' lambda', ' list', ' dict', ' tuple', ' set', ' numpy', ' pandas',
        'pyqt6', 'pyqt', 'qwidget', 'qapplication', 'qdialog', 'qlabel', 'qpixmap',
        'rag', 'vector db', 'embedding', 'chunking', 'context', 'prompt', 'llm', 'chroma',
        'collection', 'document', 'similarity', 'query', 'index',
        'my code', 'my project', 'refactor this', 'debug this', 'in my implementation',
        'according to the file', 'based on the documents', 'in these files', 'the provided context',
        'summarize this', 'search', 'find', 'lookup', 'relevant context',
        'change', 'update', 'modify',
    }
    _GREETING_PATTERNS = re.compile(r"^\s*(hi|hello|hey|yo|sup|good\s+(morning|afternoon|evening)|how\s+are\s+you)\b.*",
                                    re.IGNORECASE)
    _CODE_FENCE_PATTERN = re.compile(r"```")

    # --- NEW: Differentiated Boost Factors ---
    EXPLICIT_FOCUS_BOOST_FACTOR = 0.50  # Stronger boost (lower distance)
    IMPLICIT_FOCUS_BOOST_FACTOR = 0.70  # Moderate boost (lower distance)
    ENTITY_BOOST_FACTOR = 0.80  # Existing entity boost (lower distance)

    # --- END NEW ---

    def __init__(self, upload_service: UploadService, vector_db_service: VectorDBService):
        if not isinstance(upload_service, UploadService):
            raise TypeError("RagHandler requires a valid UploadService instance.")
        if not isinstance(vector_db_service, VectorDBService):
            raise TypeError("RagHandler requires a valid VectorDBService instance.")

        self._upload_service = upload_service
        self._vector_db_service = vector_db_service
        logger.info("RagHandler initialized.")

    def should_perform_rag(self, query: str, rag_available: bool, rag_initialized: bool) -> bool:
        """
        Determines if RAG should be performed for a given query based on heuristics.

        Args:
            query (str): The user's input query.
            rag_available (bool): True if RAG components are generally available.
            rag_initialized (bool): True if the RAG system (vector DB, etc.) is initialized.

        Returns:
            bool: True if RAG should be performed, False otherwise.
        """
        if not rag_available or not rag_initialized:
            return False
        if not query:
            return False

        query_lower = query.lower().strip()
        if len(query) < 15 and self._GREETING_PATTERNS.match(query_lower):
            return False
        if len(query) < 10:  # Short queries unlikely to need RAG unless very specific
            return False
        if self._CODE_FENCE_PATTERN.search(query):
            return True
        if any(keyword in query_lower for keyword in self._TECHNICAL_KEYWORDS):
            return True
        # Check for code-like syntax (e.g., dot notation, parentheses, brackets common in code)
        if re.search(r"[_.(){}\[\]=:]", query) and len(query) > 15:
            return True
        return False

    def extract_code_entities(self, query: str) -> Set[str]:
        """
        Extracts potential code-related entities (function names, class names, filenames)
        from a user query using regex heuristics.

        Args:
            query (str): The user's input query.

        Returns:
            Set[str]: A set of extracted code entities.
        """
        entities = set()
        if not query:
            return entities
        # Pattern to find potential function/method calls or class/function definitions
        # This is a heuristic and may capture non-code terms, but it's a starting point
        code_entity_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(\s*|\s*=\s*|\s*\.)'
        try:
            # Find words that look like they could be function/class names or variables
            for match in re.finditer(code_entity_pattern, query):
                entity = match.group(1)
                # Filter out common keywords or very short, generic words
                if len(entity) > 2 and entity.lower() not in ['def', 'class', 'self', 'init', 'str', 'repr', 'args',
                                                              'kwargs', 'return', 'true', 'false', 'none']:
                    entities.add(entity)

            # Additionally, look for direct references to filenames with extensions
            # (e.g., 'main.py', 'utils.py')
            filename_pattern = r'\b(\w+\.\w+)\b'
            for match in re.finditer(filename_pattern, query):
                entities.add(match.group(1))

        except Exception as e:
            logger.warning(f"Regex error during query entity extraction: {e}")

        if entities:
            logger.debug(f"Extracted potential code entities from query: {entities}")
        return entities

    def get_formatted_context(
            self,
            query: str,
            query_entities: Set[str],  # These are extracted by RagHandler's own method
            project_id: Optional[str],
            explicit_focus_paths: Optional[List[str]] = None,
            implicit_focus_paths: Optional[List[str]] = None,
            is_modification_request: bool = False  # Hint for retrieving more initial results
    ) -> Tuple[str, List[str]]:
        """
        Retrieves relevant context from the vector database, applies boosting heuristics,
        and formats the top results for inclusion in an LLM prompt.

        Args:
            query (str): The user's query.
            query_entities (Set[str]): Code entities extracted from the query.
            project_id (Optional[str]): The ID of the current project, if any, to query its collection.
            explicit_focus_paths (Optional[List[str]]): Paths explicitly selected by the user for focus.
            implicit_focus_paths (Optional[List[str]]): Paths implicitly focused on by recent activity.
            is_modification_request (bool): True if this is part of a code modification workflow,
                                            suggests retrieving more initial results.

        Returns:
            Tuple[str, List[str]]: A tuple containing:
                - A formatted string of the retrieved context for the LLM.
                - A list of collection IDs that were actually queried.
        """
        context_str = ""
        queried_collections_actual = []  # To track which collections were actually queried

        # Normalize focus paths for reliable comparison (absolute path and normcase for consistency)
        normalized_explicit_focus = {os.path.normcase(os.path.abspath(p)) for p in
                                     explicit_focus_paths} if explicit_focus_paths else set()
        normalized_implicit_focus = {os.path.normcase(os.path.abspath(p)) for p in
                                     implicit_focus_paths} if implicit_focus_paths else set()

        if normalized_explicit_focus: logger.info(
            f"RAG Handler: Using explicit focus paths: {normalized_explicit_focus}")
        if normalized_implicit_focus: logger.info(
            f"RAG Handler: Using implicit focus paths: {normalized_implicit_focus}")

        collections_to_query_candidates = []
        # Always try to query the global collection
        if self._vector_db_service.is_ready(GLOBAL_COLLECTION_ID):
            collections_to_query_candidates.append(GLOBAL_COLLECTION_ID)
        else:
            logger.warning(f"Global collection '{GLOBAL_COLLECTION_ID}' not ready, skipping for RAG.")

        # If a project_id is provided and its collection is ready, add it
        if project_id and project_id != GLOBAL_COLLECTION_ID and self._vector_db_service.is_ready(project_id):
            collections_to_query_candidates.append(project_id)
        elif project_id and project_id != GLOBAL_COLLECTION_ID:
            logger.warning(f"Project collection '{project_id}' not ready, skipping for RAG.")

        if not collections_to_query_candidates:
            logger.warning("RAG context requested but no ready collections to query.")
            return "", []

        logger.info(f"Attempting RAG retrieval from collections: {collections_to_query_candidates}...")
        try:
            if not hasattr(self._upload_service, 'query_vector_db'):
                raise TypeError("UploadService missing required 'query_vector_db' method.")

            # Retrieve more initial results to allow for boosting and re-ranking to be effective
            num_initial_results = constants.RAG_NUM_RESULTS * (3 if is_modification_request else 2)
            num_final_results = constants.RAG_NUM_RESULTS  # Number of chunks to return to LLM

            relevant_chunks = self._upload_service.query_vector_db(
                query,
                collection_ids=collections_to_query_candidates,
                n_results=num_initial_results
            )

            # Track which collections actually returned results
            queried_collections_actual = list(
                set(c.get("metadata", {}).get("collection_id", "N/A") for c in relevant_chunks if
                    c.get("metadata", {}).get("collection_id") != "N/A"))

            boosted_by_explicit_focus_count = 0
            boosted_by_implicit_focus_count = 0
            boosted_by_entity_count = 0

            if relevant_chunks:
                logger.debug(
                    f"Re-ranking {len(relevant_chunks)} chunks. Query Entities: {query_entities}, "
                    f"ExplicitFocus: {normalized_explicit_focus}, ImplicitFocus: {normalized_implicit_focus}")

                for chunk in relevant_chunks:
                    metadata = chunk.get('metadata')
                    # distance is already a float, no need to check type from `UploadService.query_vector_db`

                    if not isinstance(metadata, dict):
                        logger.warning(f"Skipping chunk with invalid metadata: {chunk}")
                        continue

                    boost_applied_this_chunk = False
                    chunk_source_path = metadata.get('source')
                    norm_chunk_path = None
                    if chunk_source_path:
                        try:
                            norm_chunk_path = os.path.normcase(os.path.abspath(chunk_source_path))
                        except Exception as e_norm_chunk:
                            logger.warning(
                                f"Could not normalize chunk source path '{chunk_source_path}': {e_norm_chunk}")
                            pass  # norm_chunk_path remains None

                    # 1. Check for Explicit User Focus (highest priority boost)
                    if norm_chunk_path and normalized_explicit_focus:
                        is_explicitly_focused = False
                        for focus_path in normalized_explicit_focus:
                            if os.path.isdir(focus_path):  # If the focus path is a directory
                                if norm_chunk_path.startswith(focus_path + os.sep):
                                    is_explicitly_focused = True;
                                    break
                            elif norm_chunk_path == focus_path:  # If the focus path is a specific file
                                is_explicitly_focused = True;
                                break

                        if is_explicitly_focused:
                            # Lower distance means higher relevance
                            chunk['distance'] *= self.EXPLICIT_FOCUS_BOOST_FACTOR
                            chunk['boost_reason'] = 'explicit_focus'
                            boosted_by_explicit_focus_count += 1
                            boost_applied_this_chunk = True
                            logger.debug(
                                f"  Applied EXPLICIT boost to chunk from '{chunk_source_path}'. New dist: {chunk['distance']:.4f}")

                    # 2. Check for Implicit Task Context Focus (if not already boosted by explicit)
                    if not boost_applied_this_chunk and norm_chunk_path and normalized_implicit_focus:
                        is_implicitly_focused = False
                        for focus_path in normalized_implicit_focus:
                            if os.path.isdir(focus_path):
                                if norm_chunk_path.startswith(focus_path + os.sep):
                                    is_implicitly_focused = True;
                                    break
                            elif norm_chunk_path == focus_path:
                                is_implicitly_focused = True;
                                break

                        if is_implicitly_focused:
                            chunk['distance'] *= self.IMPLICIT_FOCUS_BOOST_FACTOR
                            chunk['boost_reason'] = 'implicit_focus'
                            boosted_by_implicit_focus_count += 1
                            boost_applied_this_chunk = True
                            logger.debug(
                                f"  Applied IMPLICIT boost to chunk from '{chunk_source_path}'. New dist: {chunk['distance']:.4f}")

                    # 3. Check for Entity Boost (if not already boosted by any focus type)
                    if not boost_applied_this_chunk and query_entities and 'code_entities' in metadata:
                        chunk_entities_set = set(metadata.get('code_entities', []))  # Ensure it's a set
                        if not query_entities.isdisjoint(chunk_entities_set):  # Check for any common elements
                            chunk['distance'] *= self.ENTITY_BOOST_FACTOR
                            chunk['boost_reason'] = 'entity'
                            boosted_by_entity_count += 1
                            logger.debug(
                                f"  Applied ENTITY boost to chunk from '{chunk_source_path}'. New dist: {chunk['distance']:.4f}")

                if boosted_by_explicit_focus_count > 0 or boosted_by_implicit_focus_count > 0 or boosted_by_entity_count > 0:
                    logger.info(
                        f"Applied RAG boost: ExplicitFocus={boosted_by_explicit_focus_count}, "
                        f"ImplicitFocus={boosted_by_implicit_focus_count}, Entity={boosted_by_entity_count} chunks."
                    )

                # Re-sort chunks based on new boosted distances (lower distance is better)
                valid_chunks = [res for res in relevant_chunks if isinstance(res.get('distance'), (int, float))]
                sorted_results = sorted(valid_chunks, key=lambda x: x.get('distance', float('inf')))
                final_results = sorted_results[:num_final_results]

                context_parts = []
                retrieved_chunks_details = []
                for i, chunk in enumerate(final_results):
                    metadata = chunk.get("metadata", {})
                    filename = metadata.get("filename", "unknown_source")
                    collection_id = metadata.get("collection_id", "N/A")
                    code_content = chunk.get("content", "[Content Missing]")  # Get content from the main chunk dict
                    distance = chunk.get('distance', -1.0)
                    boost_reason = chunk.get('boost_reason')
                    start_line = metadata.get('start_line', 'N/A')
                    end_line = metadata.get('end_line', 'N/A')

                    debug_info = f"Lines {start_line}-{end_line}, Dist: {distance:.4f}"
                    if boost_reason: debug_info += f", Boost: {boost_reason}"
                    if query_entities and isinstance(metadata.get('code_entities'), list):
                        matches = query_entities.intersection(set(metadata['code_entities']))
                        if matches: debug_info += f", Matched Entities: {', '.join(matches)}"

                    # Heuristic for highlighting: if content has many newlines or code markers
                    # we can assume it's a code block and format it with ```python
                    # Otherwise, use ```markdown
                    is_code_like = '\n' in code_content or any(kw in code_content.lower() for kw in ['def ', 'class '])
                    fenced_lang = 'python' if is_code_like or filename.endswith('.py') else 'markdown'

                    context_parts.append(
                        f"--- Snippet {i + 1} from `{filename}` (Collection: {collection_id}) ({debug_info}) ---\n"
                        f"```{fenced_lang}\n"
                        f"{code_content}\n"
                        f"```\n")
                    retrieved_chunks_details.append(f"{filename} {debug_info}")

                if context_parts:
                    context_str = ("--- Relevant Code Context Start ---\n" + "\n".join(
                        context_parts) + "--- Relevant Code Context End ---")
                    logger.info(
                        f"Final RAG context includes {len(final_results)} chunks: [{', '.join(retrieved_chunks_details)}]")
                else:
                    logger.info("No valid chunks remained after processing/sorting.")
            else:
                logger.info(f"No relevant RAG context found in collections {collections_to_query_candidates}.")

        except Exception as e_rag:
            logger.exception("Error retrieving/re-ranking RAG context:")
            context_str = "[Error retrieving RAG context]"

        return context_str, queried_collections_actual