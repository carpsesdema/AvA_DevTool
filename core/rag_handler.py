# core/rag_handler.py
import logging
import os
import re
from typing import List, Optional, Set, Tuple, Dict, Any

try:
    from services.upload_service import UploadService
    from services.vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID
    from utils import constants
except ImportError as e:
    logging.critical(f"RagHandler: Failed to import services/utils: {e}")
    UploadService = type("UploadService", (object,), {})
    VectorDBService = type("VectorDBService", (object,), {})
    GLOBAL_COLLECTION_ID = "global_knowledge_fallback"  # Fallback
    constants = type("constants", (object,),
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

    EXPLICIT_FOCUS_BOOST_FACTOR = 0.50
    IMPLICIT_FOCUS_BOOST_FACTOR = 0.70
    ENTITY_BOOST_FACTOR = 0.80

    def __init__(self, upload_service: Optional[UploadService], vector_db_service: Optional[VectorDBService]):
        self._upload_service = None
        self._vector_db_service = None

        if not upload_service or not isinstance(upload_service, UploadService):
            logger.warning("RagHandler initialized with invalid or missing UploadService")
        else:
            self._upload_service = upload_service

        if not vector_db_service or not isinstance(vector_db_service, VectorDBService):
            logger.warning("RagHandler initialized with invalid or missing VectorDBService")
        else:
            self._vector_db_service = vector_db_service

        logger.info("RagHandler initialized with available services: "
                    f"UploadService={self._upload_service is not None}, "
                    f"VectorDBService={self._vector_db_service is not None}")

    def should_perform_rag(self, query: str, rag_available: bool, rag_initialized: bool) -> bool:
        if not rag_available or not rag_initialized:
            return False
        if not query:
            return False
        query_lower = query.lower().strip()
        if len(query) < 15 and self._GREETING_PATTERNS.match(query_lower):
            return False
        if len(query) < 10:
            return False
        if self._CODE_FENCE_PATTERN.search(query):
            return True
        if any(keyword in query_lower for keyword in self._TECHNICAL_KEYWORDS):
            return True
        if re.search(r"[_.(){}\[\]=:]", query) and len(query) > 15:
            return True
        return False

    def extract_code_entities(self, query: str) -> Set[str]:
        entities = set()
        if not query:
            return entities
        code_entity_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(\s*|\s*=\s*|\s*\.)'
        try:
            for match in re.finditer(code_entity_pattern, query):
                entity = match.group(1)
                if len(entity) > 2 and entity.lower() not in ['def', 'class', 'self', 'init', 'str', 'repr', 'args',
                                                              'kwargs', 'return', 'true', 'false', 'none']:
                    entities.add(entity)
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
            query_entities: Set[str],
            project_id: Optional[str],
            explicit_focus_paths: Optional[List[str]] = None,
            implicit_focus_paths: Optional[List[str]] = None,
            is_modification_request: bool = False
    ) -> Tuple[str, List[str]]:
        if not self._upload_service or not self._vector_db_service:
            logger.warning("RAG services unavailable - cannot retrieve context")
            return "", []

        context_str = ""
        queried_collections_actual = []

        normalized_explicit_focus = {os.path.normcase(os.path.abspath(p)) for p in
                                     explicit_focus_paths} if explicit_focus_paths else set()
        normalized_implicit_focus = {os.path.normcase(os.path.abspath(p)) for p in
                                     implicit_focus_paths} if implicit_focus_paths else set()

        if normalized_explicit_focus: logger.info(
            f"RAG Handler: Using explicit focus paths: {normalized_explicit_focus}")
        if normalized_implicit_focus: logger.info(
            f"RAG Handler: Using implicit focus paths: {normalized_implicit_focus}")

        collections_to_query_candidates: List[str] = []

        # 1. Add project-specific collection if project_id is provided and it's ready
        if project_id and project_id != GLOBAL_COLLECTION_ID:  # Ensure project_id is not the global one
            if self._vector_db_service.is_ready(project_id):
                collections_to_query_candidates.append(project_id)
                logger.info(f"RAG: Added project collection '{project_id}' to query candidates.")
            else:
                logger.warning(f"RAG: Project collection '{project_id}' is not ready, skipping.")

        # 2. Always add GLOBAL_COLLECTION_ID if it's ready (and not already added if project_id was global)
        if GLOBAL_COLLECTION_ID not in collections_to_query_candidates:
            if self._vector_db_service.is_ready(GLOBAL_COLLECTION_ID):
                collections_to_query_candidates.append(GLOBAL_COLLECTION_ID)
                logger.info(f"RAG: Added global collection '{GLOBAL_COLLECTION_ID}' to query candidates.")
            else:
                logger.warning(f"RAG: Global collection '{GLOBAL_COLLECTION_ID}' not ready, skipping.")

        # Ensure uniqueness if somehow GLOBAL_COLLECTION_ID was also the project_id
        collections_to_query_candidates = list(dict.fromkeys(collections_to_query_candidates))

        if not collections_to_query_candidates:
            logger.warning("RAG context requested but no ready collections to query.")
            return "", []

        logger.info(f"Attempting RAG retrieval from collections: {collections_to_query_candidates}...")
        try:
            if not hasattr(self._upload_service, 'query_vector_db'):
                logger.error("UploadService missing required 'query_vector_db' method.")
                return "", []

            num_initial_results = getattr(constants, 'RAG_NUM_RESULTS', 5) * (3 if is_modification_request else 2)
            num_final_results = getattr(constants, 'RAG_NUM_RESULTS', 5)

            # UploadService.query_vector_db now queries all listed collections and returns combined results
            relevant_chunks = self._upload_service.query_vector_db(
                query,
                collection_ids=collections_to_query_candidates,  # Pass all candidates
                n_results=num_initial_results
            )

            # Track which collections actually returned results based on metadata from UploadService
            queried_collections_actual = list(
                set(c.get("metadata", {}).get("retrieved_from_collection", "N/A") for c in relevant_chunks if
                    c.get("metadata", {}).get("retrieved_from_collection") != "N/A"))

            boosted_by_explicit_focus_count = 0
            boosted_by_implicit_focus_count = 0
            boosted_by_entity_count = 0

            if relevant_chunks:
                logger.debug(
                    f"Re-ranking {len(relevant_chunks)} chunks. Query Entities: {query_entities}, "
                    f"ExplicitFocus: {normalized_explicit_focus}, ImplicitFocus: {normalized_implicit_focus}")

                for chunk in relevant_chunks:
                    metadata = chunk.get('metadata')
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
                            pass

                    if norm_chunk_path and normalized_explicit_focus:
                        is_explicitly_focused = False
                        for focus_path in normalized_explicit_focus:
                            if os.path.isdir(focus_path):
                                if norm_chunk_path.startswith(focus_path + os.sep):
                                    is_explicitly_focused = True;
                                    break
                            elif norm_chunk_path == focus_path:
                                is_explicitly_focused = True;
                                break
                        if is_explicitly_focused:
                            chunk['distance'] *= self.EXPLICIT_FOCUS_BOOST_FACTOR
                            chunk['boost_reason'] = 'explicit_focus'
                            boosted_by_explicit_focus_count += 1
                            boost_applied_this_chunk = True
                            logger.debug(
                                f"  Applied EXPLICIT boost to chunk from '{chunk_source_path}'. New dist: {chunk['distance']:.4f}")

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

                    if not boost_applied_this_chunk and query_entities and 'code_entities' in metadata:
                        chunk_entities_set = set(
                            str(metadata.get('code_entities', "")).split(", "))  # Handle string format from DB
                        if "" in chunk_entities_set: chunk_entities_set.remove("")  # Remove empty strings if any
                        if not query_entities.isdisjoint(chunk_entities_set):
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

                valid_chunks = [res for res in relevant_chunks if isinstance(res.get('distance'), (int, float))]
                sorted_results = sorted(valid_chunks, key=lambda x: x.get('distance', float('inf')))
                final_results = sorted_results[:num_final_results]

                context_parts = []
                retrieved_chunks_details = []
                for i, chunk in enumerate(final_results):
                    metadata = chunk.get("metadata", {})
                    filename = metadata.get("filename", "unknown_source")
                    # Use 'retrieved_from_collection' if available, else original 'collection_id'
                    collection_id_display = metadata.get("retrieved_from_collection",
                                                         metadata.get("collection_id", "N/A"))
                    code_content = chunk.get("content", "[Content Missing]")
                    distance = chunk.get('distance', -1.0)
                    boost_reason = chunk.get('boost_reason')
                    start_line = metadata.get('start_line', 'N/A')
                    end_line = metadata.get('end_line', 'N/A')

                    debug_info = f"Lines {start_line}-{end_line}, Dist: {distance:.4f}"
                    if boost_reason: debug_info += f", Boost: {boost_reason}"

                    # Handling code_entities which is now a string from DB
                    chunk_code_entities_str = metadata.get('code_entities', "")
                    if query_entities and chunk_code_entities_str:
                        chunk_entities_set = set(
                            ent.strip() for ent in chunk_code_entities_str.split(',') if ent.strip())
                        matches = query_entities.intersection(chunk_entities_set)
                        if matches: debug_info += f", Matched Entities: {', '.join(matches)}"

                    is_code_like = '\n' in code_content or any(kw in code_content.lower() for kw in ['def ', 'class '])
                    fenced_lang = 'python' if is_code_like or filename.endswith('.py') else 'markdown'

                    context_parts.append(
                        f"--- Snippet {i + 1} from `{filename}` (Collection: {collection_id_display}) ({debug_info}) ---\n"
                        f"```{fenced_lang}\n"
                        f"{code_content}\n"
                        f"```\n")
                    retrieved_chunks_details.append(f"{filename} ({collection_id_display}) {debug_info}")

                if context_parts:
                    context_str = ("--- Relevant Code Context Start ---\n" + "\n".join(
                        context_parts) + "--- Relevant Code Context End ---")
                    logger.info(
                        f"Final RAG context includes {len(final_results)} chunks from {len(queried_collections_actual)} collection(s): [{', '.join(retrieved_chunks_details)}]")
                else:
                    logger.info("No valid chunks remained after processing/sorting.")
            else:
                logger.info(f"No relevant RAG context found in collections {collections_to_query_candidates}.")

        except Exception as e_rag:
            logger.exception(f"Error retrieving/re-ranking RAG context: {e_rag}")
            context_str = "[Error retrieving RAG context]"

        return context_str, queried_collections_actual