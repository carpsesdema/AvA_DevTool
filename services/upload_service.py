# services/upload_service.py
import asyncio
import datetime
import logging
import os
import sys
from html import escape
from typing import List, Tuple, Optional, Set, Dict, Any

import numpy as np  # Keep this import, it's used

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    SentenceTransformer = None  # type: ignore
    EMBEDDINGS_AVAILABLE = False
    DEFAULT_EMBEDDING_MODEL = "fallback_model_name"  # Define for logging
    logging.error(f"UploadService: SentenceTransformer library import failed ({e}). RAG will likely fail.",
                  exc_info=True)

try:
    import numpy  # Ensure numpy is imported if not already (it's in the original)

    NUMPY_AVAILABLE = True
except ImportError:
    numpy = None  # type: ignore
    NUMPY_AVAILABLE = False
    logging.error("UploadService: Numpy library not found. RAG DB cannot function.", exc_info=True)

from utils import constants
from core.models import SYSTEM_ROLE, ChatMessage, ERROR_ROLE

try:
    from services.chunking_service import ChunkingService

    CHUNKING_SERVICE_AVAILABLE = True
except ImportError as e:
    ChunkingService = None  # type: ignore
    CHUNKING_SERVICE_AVAILABLE = False
    logging.error(
        f"UploadService: Failed to import ChunkingService ({e}).", exc_info=True)
try:
    from services.vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID

    VECTOR_DB_SERVICE_AVAILABLE = True
except ImportError as e:
    VectorDBService = None  # type: ignore
    VECTOR_DB_SERVICE_AVAILABLE = False
    logging.error(
        f"UploadService: Failed to import VectorDBService ({e}).", exc_info=True)
    GLOBAL_COLLECTION_ID = "global_knowledge_fallback"  # Fallback if import fails
try:
    from services.file_handler_service import FileHandlerService

    FILE_HANDLER_SERVICE_AVAILABLE = True
except ImportError as e:
    FileHandlerService = None  # type: ignore
    FILE_HANDLER_SERVICE_AVAILABLE = False
    logging.error(
        f"UploadService: Failed to import FileHandlerService ({e}).", exc_info=True)
try:
    from services.code_analysis_service import CodeAnalysisService

    CODE_ANALYSIS_SERVICE_AVAILABLE = True
except ImportError as e:
    CodeAnalysisService = None  # type: ignore
    CODE_ANALYSIS_SERVICE_AVAILABLE = False
    logging.error(
        f"UploadService: Failed to import CodeAnalysisService ({e}).", exc_info=True)

logger = logging.getLogger(__name__)
CHROMA_DB_UPLOAD_BATCH_SIZE = 4000


class UploadService:
    def __init__(self):
        logger.info("UploadService initializing...")
        self._embedder: Optional[SentenceTransformer] = None
        self._chunking_service: Optional[ChunkingService] = None
        self._vector_db_service: Optional[VectorDBService] = None
        self._file_handler_service: Optional[FileHandlerService] = None
        self._code_analysis_service: Optional[CodeAnalysisService] = None
        self._index_dim = -1
        self._dependencies_ready = False
        self._embedder_init_task: Optional[asyncio.Task] = None
        self._embedder_ready = False

        # AVA_ASSISTANT_MODIFIED: Enhanced logging for dependency check
        logger.info(f"  Dependency Status: EMBEDDINGS_AVAILABLE={EMBEDDINGS_AVAILABLE}")
        logger.info(f"  Dependency Status: CHUNKING_SERVICE_AVAILABLE={CHUNKING_SERVICE_AVAILABLE}")
        logger.info(f"  Dependency Status: VECTOR_DB_SERVICE_AVAILABLE={VECTOR_DB_SERVICE_AVAILABLE}")
        logger.info(f"  Dependency Status: FILE_HANDLER_SERVICE_AVAILABLE={FILE_HANDLER_SERVICE_AVAILABLE}")
        logger.info(f"  Dependency Status: CODE_ANALYSIS_SERVICE_AVAILABLE={CODE_ANALYSIS_SERVICE_AVAILABLE}")
        logger.info(f"  Dependency Status: NUMPY_AVAILABLE={NUMPY_AVAILABLE}")

        if not all([EMBEDDINGS_AVAILABLE, CHUNKING_SERVICE_AVAILABLE, VECTOR_DB_SERVICE_AVAILABLE,
                    FILE_HANDLER_SERVICE_AVAILABLE, CODE_ANALYSIS_SERVICE_AVAILABLE, NUMPY_AVAILABLE]):
            logger.critical("UploadService cannot initialize due to missing critical dependencies. Check logs above.")
            return

        try:
            # Initialize non-embedding services first
            self._file_handler_service = FileHandlerService()
            logger.info("FileHandlerService initialized.")

            self._chunking_service = ChunkingService(
                chunk_size=getattr(constants, 'RAG_CHUNK_SIZE', 1000),
                chunk_overlap=getattr(constants, 'RAG_CHUNK_OVERLAP', 150)
            )
            logger.info("ChunkingService initialized.")

            self._code_analysis_service = CodeAnalysisService()
            logger.info("CodeAnalysisService initialized.")

            # Initialize VectorDB with a default dimension (will be updated when embedder is ready)
            self._vector_db_service = VectorDBService(index_dimension=384)  # Default for all-MiniLM-L6-v2
            logger.info("VectorDBService initialized with default dimension.")

            # Start embedder initialization in background
            self._start_embedder_initialization()

        except Exception as e:
            logger.exception(f"CRITICAL FAILURE during UploadService component initialization: {e}")
            self._dependencies_ready = False

    def _start_embedder_initialization(self):
        """Start embedder initialization in background"""
        if not EMBEDDINGS_AVAILABLE:
            logger.error("Cannot initialize embedder: SentenceTransformers not available")
            return

        logger.info("Starting background embedder initialization...")
        self._embedder_init_task = asyncio.create_task(self._initialize_embedder_async())

    async def _initialize_embedder_async(self):
        """Initialize embedder asynchronously to avoid blocking the UI"""
        try:
            logger.info("Initializing SentenceTransformer in background...")

            # Run the potentially blocking operation in a thread pool
            loop = asyncio.get_event_loop()
            self._embedder = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            )

            if self._embedder:
                logger.info("SentenceTransformer initialized successfully.")

                # Test embedding and get dimension
                test_embedding = await loop.run_in_executor(
                    None,
                    lambda: self._embedder.encode(["test"])
                )

                if test_embedding is not None and hasattr(test_embedding, 'shape') and len(test_embedding.shape) > 1:
                    self._index_dim = test_embedding.shape[1]
                    logger.info(f"Detected embedding dimension: {self._index_dim}")

                    # Update VectorDB service with correct dimension
                    if self._vector_db_service:
                        self._vector_db_service._index_dim = self._index_dim

                    self._embedder_ready = True
                    self._dependencies_ready = self._vector_db_service.is_ready() if self._vector_db_service else False

                    if self._dependencies_ready:
                        logger.info("UploadService fully ready after background initialization.")
                    else:
                        logger.error("VectorDBService not ready after embedder initialization.")
                else:
                    logger.error("Failed to get valid embedding shape after SentenceTransformer init.")
                    raise ValueError("Failed to get valid embedding shape.")
            else:
                logger.error("SentenceTransformer failed to initialize (returned None).")
                raise ValueError("SentenceTransformer failed to initialize.")

        except Exception as e:
            logger.exception(f"Failed to initialize embedder in background: {e}")
            self._embedder_ready = False
            self._dependencies_ready = False

    def is_vector_db_ready(self, collection_id: Optional[str] = None) -> bool:
        if not self._embedder_ready or not self._dependencies_ready or not self._vector_db_service:
            logger.debug(
                f"is_vector_db_ready check: Embedder ready={self._embedder_ready}, dependencies ready={self._dependencies_ready}")
            return False

        if collection_id is None:
            # General check for client readiness
            client_ready = self._vector_db_service.is_ready()
            logger.debug(f"is_vector_db_ready check (general client): {client_ready}")
            return client_ready

        try:
            # Specific collection check
            collection = self._vector_db_service.get_or_create_collection(collection_id)
            collection_exists_and_accessible = collection is not None
            logger.debug(
                f"is_vector_db_ready check for collection '{collection_id}': {collection_exists_and_accessible}")
            return collection_exists_and_accessible
        except Exception as e:
            logger.warning(f"Error checking readiness for collection '{collection_id}': {e}", exc_info=True)
            return False

    async def wait_for_embedder_ready(self, timeout_seconds: float = 30.0) -> bool:
        """Wait for embedder to be ready, with timeout"""
        if self._embedder_ready:
            return True

        if not self._embedder_init_task:
            logger.error("No embedder initialization task running")
            return False

        try:
            await asyncio.wait_for(self._embedder_init_task, timeout=timeout_seconds)
            return self._embedder_ready
        except asyncio.TimeoutError:
            logger.error(f"Embedder initialization timed out after {timeout_seconds} seconds")
            return False
        except Exception as e:
            logger.error(f"Error waiting for embedder: {e}")
            return False

    def _send_batch_to_db(self, collection_id: str,
                          batch_contents: List[str],
                          batch_embeddings: List[List[float]],
                          batch_metadatas: List[Dict[str, Any]],
                          files_in_this_batch_names: Set[str]) -> Tuple[bool, Set[str], int]:
        if not batch_contents: return True, set(), 0
        if not self._vector_db_service:
            logger.error("UploadService: _vector_db_service is None in _send_batch_to_db.")
            return False, set(), 0

        num_embeddings_in_batch = len(batch_embeddings)
        try:
            logger.info(
                f"Sending batch of {len(batch_contents)} docs from {len(files_in_this_batch_names)} files to coll '{collection_id}'...")
            success = self._vector_db_service.add_embeddings(collection_id, batch_contents, batch_embeddings,
                                                             batch_metadatas)
            if success:
                logger.info(
                    f"Successfully added {len(batch_contents)} documents ({num_embeddings_in_batch} embeddings) to '{collection_id}'.")
                return True, files_in_this_batch_names, num_embeddings_in_batch
            else:
                logger.error(
                    f"DB service reported failure for batch to coll '{collection_id}'. Files: {files_in_this_batch_names}")
                return False, set(), num_embeddings_in_batch  # Return 0 for successful embeddings on failure
        except Exception as e:
            logger.exception(
                f"Exception during batch add to coll '{collection_id}': {e}. Files: {files_in_this_batch_names}")
            return False, set(), num_embeddings_in_batch  # Return 0 for successful embeddings on failure

    async def process_files_for_context_async(self, file_paths: List[str], collection_id: str) -> Optional[ChatMessage]:
        """Async version of process_files_for_context that waits for embedder"""
        if not await self.wait_for_embedder_ready():
            return ChatMessage(role=ERROR_ROLE, parts=["[Error: RAG embedder not ready. Please try again.]"])

        return self.process_files_for_context(file_paths, collection_id)

    def process_files_for_context(self, file_paths: List[str], collection_id: str) -> Optional[ChatMessage]:
        if not collection_id:
            logger.error("UploadService: process_files_for_context called without a collection_id.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: Collection ID for RAG processing is missing.]"])

        if not isinstance(file_paths, list):
            return ChatMessage(role=ERROR_ROLE, parts=["[System Error: Invalid input provided.]"])

        num_input_files = len(file_paths)
        logger.info(f"Processing {num_input_files} files for RAG collection '{collection_id}'...")

        # Check if embedder is ready
        if not self._embedder_ready or not self._embedder:
            logger.error("UploadService: Embedder not ready for file processing.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[Error: RAG embedder not ready. Please wait for initialization to complete.]"])

        overall_successfully_added_files = set()
        processing_error_files_dict = {}
        db_failed_files_exclusive_set = set()
        binary_skipped_files = set()
        no_content_files = set()

        total_chunks_generated = 0
        total_embeddings_in_successful_batches = 0
        total_embeddings_in_failed_batches = 0
        any_db_batch_add_failure_occurred = False

        # Check dependencies readiness
        if not self._dependencies_ready:
            logger.error("UploadService: Core dependencies not ready. Cannot process files for RAG.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[Error: RAG core components failed to initialize. Cannot process files.]"])

        if not self.is_vector_db_ready(collection_id):
            logger.error(f"UploadService: Vector DB/collection '{collection_id}' not ready. Cannot process files.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=[f"[Error: RAG DB/collection '{collection_id}' not ready.]"])

        if not self._embedder or not self._file_handler_service or not self._chunking_service or not self._code_analysis_service:
            logger.error(
                "UploadService: One or more internal services (embedder, file_handler, chunking, code_analysis) are None.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: RAG processing sub-components not ready.]"])

        current_batch_contents: List[str] = []
        current_batch_embeddings: List[List[float]] = []
        current_batch_metadatas: List[Dict[str, Any]] = []
        current_batch_file_names_involved: Set[str] = set()

        for i, file_path in enumerate(file_paths):
            display_name = os.path.basename(file_path)
            if not os.path.exists(file_path): processing_error_files_dict[escape(display_name)] = "Not Found"; continue
            if not os.path.isfile(file_path): processing_error_files_dict[escape(display_name)] = "Not a File"; continue

            logger.info(f"  Processing [{i + 1}/{num_input_files}]: {display_name} for collection '{collection_id}'")
            read_result = self._file_handler_service.read_file_content(file_path)
            if not read_result or len(read_result) != 3: processing_error_files_dict[
                escape(display_name)] = "Internal Read Error"; continue
            content, file_type, error_msg = read_result
            if file_type == "error": processing_error_files_dict[
                escape(display_name)] = error_msg or "Read Error"; continue
            if file_type == "binary": binary_skipped_files.add(escape(display_name)); continue

            code_structures = []
            if os.path.splitext(file_path)[1].lower() == '.py' and self._code_analysis_service:
                try:
                    code_structures = self._code_analysis_service.parse_python_structures(content if content else "",
                                                                                          file_path)
                except Exception as e_code_parse:
                    logger.warning(f"Error parsing code structures for {display_name}: {e_code_parse}", exc_info=True)
                    processing_error_files_dict[
                        escape(display_name)] = "Code Parse Error"  # Still try to chunk raw content

            if escape(display_name) in processing_error_files_dict and "Code Parse Error" in \
                    processing_error_files_dict[escape(display_name)]:
                pass  # Allow chunking even if code parsing had an issue
            elif escape(display_name) in processing_error_files_dict:
                continue

            if not content:
                no_content_files.add(escape(display_name))
                continue

            try:
                chunks = self._chunking_service.chunk_document(content, source_id=file_path,
                                                               file_ext=os.path.splitext(file_path)[1].lower())
                if not chunks:
                    if content.strip(): no_content_files.add(escape(display_name))
                    continue

                total_chunks_generated += len(chunks)
                file_specific_chunk_contents: List[str] = []
                file_specific_metadatas: List[Dict[str, Any]] = []

                for chunk_data in chunks:
                    if not isinstance(chunk_data,
                                      dict) or 'metadata' not in chunk_data or 'content' not in chunk_data: continue
                    md = chunk_data['metadata']
                    entities = [s["name"] for s in code_structures if
                                s.get("start_line", -1) <= md.get('end_line', 0) and s.get("end_line", 0) >= md.get(
                                    'start_line', -1) and s.get("name")]
                    md['code_entities'] = ", ".join(sorted(list(set(entities)))) if entities else ""
                    md['collection_id'] = collection_id
                    file_specific_chunk_contents.append(chunk_data['content'])
                    file_specific_metadatas.append(md)

                if not file_specific_chunk_contents: no_content_files.add(escape(display_name)); continue

                embeddings_np = self._embedder.encode(file_specific_chunk_contents, show_progress_bar=False)
                if not isinstance(embeddings_np, np.ndarray) or embeddings_np.ndim != 2:
                    logger.error(f"Embedder did not return a 2D numpy array for {display_name}. Skipping file.")
                    processing_error_files_dict[escape(display_name)] = "Embedding Error"
                    continue

                current_batch_contents.extend(file_specific_chunk_contents)
                current_batch_embeddings.extend(embeddings_np.tolist())
                current_batch_metadatas.extend(file_specific_metadatas)
                current_batch_file_names_involved.add(escape(display_name))

                if len(current_batch_contents) >= CHROMA_DB_UPLOAD_BATCH_SIZE:
                    batch_ok, files_added_names, embs_count = self._send_batch_to_db(collection_id,
                                                                                     current_batch_contents,
                                                                                     current_batch_embeddings,
                                                                                     current_batch_metadatas,
                                                                                     current_batch_file_names_involved)
                    if batch_ok:
                        overall_successfully_added_files.update(files_added_names)
                        total_embeddings_in_successful_batches += embs_count
                    else:
                        any_db_batch_add_failure_occurred = True
                        total_embeddings_in_failed_batches += embs_count
                        for f_name_failed in current_batch_file_names_involved:
                            if f_name_failed not in processing_error_files_dict:
                                db_failed_files_exclusive_set.add(f_name_failed)
                    current_batch_contents, current_batch_embeddings, current_batch_metadatas, current_batch_file_names_involved = [], [], [], set()
            except Exception as e_proc:
                logger.exception(f"Error processing file {display_name} for collection '{collection_id}': {e_proc}")
                processing_error_files_dict[escape(display_name)] = "Processing Error"

        if current_batch_contents:
            batch_ok, files_added_names, embs_count = self._send_batch_to_db(collection_id, current_batch_contents,
                                                                             current_batch_embeddings,
                                                                             current_batch_metadatas,
                                                                             current_batch_file_names_involved)
            if batch_ok:
                overall_successfully_added_files.update(files_added_names)
                total_embeddings_in_successful_batches += embs_count
            else:
                any_db_batch_add_failure_occurred = True
                total_embeddings_in_failed_batches += embs_count
                for f_name_failed in current_batch_file_names_involved:
                    if f_name_failed not in processing_error_files_dict:
                        db_failed_files_exclusive_set.add(f_name_failed)

        if num_input_files == 0: return ChatMessage(role=SYSTEM_ROLE,
                                                    parts=[
                                                        f"[RAG Upload: No files provided for collection '{collection_id}'.]"])

        status_notes = []
        if overall_successfully_added_files:
            status_notes.append(
                f"{len(overall_successfully_added_files)} file(s) added ({total_embeddings_in_successful_batches} embeddings)")

        if db_failed_files_exclusive_set:
            status_notes.append(
                f"{len(db_failed_files_exclusive_set)} file(s) failed DB add ({total_embeddings_in_failed_batches} attempted embeddings)")
        elif any_db_batch_add_failure_occurred and not overall_successfully_added_files:
            status_notes.append(
                f"All DB batches failed ({total_embeddings_in_failed_batches} attempted embeddings)")

        if processing_error_files_dict: status_notes.append(
            f"{len(processing_error_files_dict)} file(s) with processing errors")
        if binary_skipped_files: status_notes.append(f"{len(binary_skipped_files)} binary file(s) ignored")

        actual_no_content_files = no_content_files - set(processing_error_files_dict.keys())
        if actual_no_content_files: status_notes.append(
            f"{len(actual_no_content_files)} file(s) yielded no RAG content")

        message_role = SYSTEM_ROLE
        if processing_error_files_dict or any_db_batch_add_failure_occurred: message_role = ERROR_ROLE

        summary_parts = [f"RAG Upload: Processed {num_input_files} item(s) for Collection ID '{collection_id}'."]
        if status_notes:
            summary_parts.append("Summary: " + "; ".join(status_notes) + ".")
        else:
            summary_parts.append("Summary: Processing complete. No items added or errors noted (check logs).")

        issue_details_list = [f"'{fname} ({reason})'" for fname, reason in processing_error_files_dict.items()]
        for fname_db_failed in db_failed_files_exclusive_set:
            issue_details_list.append(f"'{escape(fname_db_failed)} (DB Batch Add Failed)'")

        if issue_details_list:
            issue_details_str = ", ".join(sorted(list(set(issue_details_list))))
            if len(issue_details_str) > 250: issue_details_str = issue_details_str[:247] + "...'"
            summary_parts.append(f"Issue details: {issue_details_str}")

        summary_text = " ".join(summary_parts)
        logger.info(f"UploadService: Finished processing for collection '{collection_id}'. Final: {summary_text}")
        return ChatMessage(role=message_role, parts=[summary_text], timestamp=datetime.datetime.now().isoformat(),
                           metadata={
                               "upload_summary_v5": {
                                   "input_files": num_input_files,
                                   "successfully_added_files": len(overall_successfully_added_files),
                                   "embeddings_in_successful_batches": total_embeddings_in_successful_batches,
                                   "processing_error_files": len(processing_error_files_dict),
                                   "db_failed_files_exclusive": len(db_failed_files_exclusive_set),
                                   "embeddings_in_failed_db_batches": total_embeddings_in_failed_batches,
                                   "binary_skipped_files": len(binary_skipped_files),
                                   "no_content_files": len(actual_no_content_files),
                                   "collection_id": collection_id,
                               }, "total_chunks_generated": total_chunks_generated
                           })

    def process_directory_for_context(self, dir_path: str, collection_id: str) -> Optional[ChatMessage]:
        logger.info(f"Processing directory '{dir_path}' for RAG collection '{collection_id}'")
        if not collection_id:
            logger.error("UploadService: process_directory_for_context called without a collection_id.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: Collection ID for RAG processing is missing.]"])
        try:
            if not os.path.isdir(dir_path): return ChatMessage(role=ERROR_ROLE, parts=[
                f"[Error: Not a directory: '{os.path.basename(dir_path)}']"])

            if not self._file_handler_service:
                logger.error("UploadService: FileHandlerService not initialized in process_directory_for_context.")
                return ChatMessage(role=ERROR_ROLE,
                                   parts=["[System Error: File handling service not ready.]"])

            valid_files, skipped_scan_info = self._scan_directory(dir_path)
            if not valid_files:
                msg = f"[RAG Scan: No suitable files in '{os.path.basename(dir_path)}' for collection '{collection_id}'."
                if skipped_scan_info:
                    msg += f" {len(skipped_scan_info)} items skipped/errored during scan.]"
                else:
                    msg += "]"
                return ChatMessage(role=SYSTEM_ROLE, parts=[msg])

            process_msg_obj = self.process_files_for_context(valid_files, collection_id=collection_id)
            if process_msg_obj and skipped_scan_info:
                scan_issue_txt = f"{len(skipped_scan_info)} items skipped/error in dir scan."
                if process_msg_obj.metadata: process_msg_obj.metadata[
                    "dir_scan_issues"] = scan_issue_txt
                if process_msg_obj.parts and isinstance(process_msg_obj.parts[0], str):
                    process_msg_obj.parts[0] = process_msg_obj.parts[0].rstrip(
                        ' .]') + f" | DirScan: {scan_issue_txt}]"
            return process_msg_obj
        except Exception as e:
            logger.exception(f"CRITICAL ERROR processing directory '{dir_path}' for collection '{collection_id}': {e}")
            return ChatMessage(role=ERROR_ROLE,
                               parts=[
                                   f"[System: Critical error processing directory '{os.path.basename(dir_path)}' for collection '{collection_id}'.]"])

    def query_vector_db(self, query_text: str, collection_ids: List[str],
                        n_results: int = constants.RAG_NUM_RESULTS) -> List[Dict[str, Any]]:
        if not self._embedder_ready or not self._embedder or not self._vector_db_service:
            logger.warning("query_vector_db: Core dependencies not ready. Returning empty list.")
            return []
        if not query_text.strip(): return []
        n_results = max(1, n_results)
        if not collection_ids:
            logger.warning(
                "UploadService.query_vector_db called with empty collection_ids, defaulting to GLOBAL_COLLECTION_ID.")
            collection_ids = [GLOBAL_COLLECTION_ID]

        all_results: List[Dict[str, Any]] = []
        try:
            query_embedding = self._embedder.encode([query_text]).tolist()
            if not query_embedding or not query_embedding[0] or len(query_embedding[0]) != self._index_dim:
                logger.error(
                    f"Failed to generate valid query embedding or dimension mismatch. Expected {self._index_dim}, got {len(query_embedding[0]) if query_embedding and query_embedding[0] else 'N/A'}")
                return []

            for coll_id in collection_ids:
                if not coll_id or not isinstance(coll_id, str):
                    logger.warning(f"Skipping invalid collection_id in query: {coll_id}")
                    continue
                if self.is_vector_db_ready(coll_id):
                    logger.debug(f"Querying collection: {coll_id} for '{query_text[:30]}...'")
                    coll_results = self._vector_db_service.search(coll_id, query_embedding, k=n_results)
                    if coll_results:
                        for res_item in coll_results:
                            if 'metadata' not in res_item or res_item['metadata'] is None:
                                res_item['metadata'] = {}
                            res_item['metadata']['retrieved_from_collection'] = coll_id
                        all_results.extend(coll_results)
                else:
                    logger.warning(f"Collection '{coll_id}' not ready, skipping in query.")
            return all_results
        except Exception as e:
            logger.exception(f"Error querying Vector DB: {e}")
            return []

    def _scan_directory(self, root_dir: str, allowed_extensions: Set[str] = constants.ALLOWED_TEXT_EXTENSIONS,
                        ignored_dirs: Set[str] = constants.DEFAULT_IGNORED_DIRS) -> Tuple[
        List[str], List[str]]:
        valid_files, skipped_info = [], []
        logger.info(f"Scanning directory: {root_dir}")
        ignored_dirs_lower = {d.lower() for d in ignored_dirs if isinstance(d, str)}
        allowed_extensions_lower = {e.lower() for e in allowed_extensions if isinstance(e, str)}
        max_depth = getattr(constants, 'MAX_SCAN_DEPTH', 5)
        max_size_mb = getattr(constants, 'RAG_MAX_FILE_SIZE_MB', 50)
        max_size_bytes = max_size_mb * 1024 * 1024
        if not os.path.isdir(root_dir):
            skipped_info.append(f"Error: Root path is not a directory: {root_dir}")
            return [], skipped_info
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True, onerror=None):
                try:
                    rel_dirpath = os.path.relpath(dirpath, root_dir)
                except ValueError:
                    logger.warning(f"Cannot get relative path for {dirpath} from {root_dir}. Assuming depth 0.")
                    depth = 0
                    rel_dirpath = dirpath

                if rel_dirpath != '.':
                    depth = rel_dirpath.count(os.sep)
                else:
                    depth = 0

                if depth >= max_depth: skipped_info.append(f"Max Depth ({max_depth}): '{rel_dirpath}'"); dirnames[
                                                                                                         :] = []; continue

                original_dirnames = list(dirnames)
                dirnames[:] = []
                for d in original_dirnames:
                    if d.startswith('.') or d.lower() in ignored_dirs_lower:
                        skipped_info.append(f"Ignored Dir: '{os.path.join(rel_dirpath, d)}'")
                    else:
                        dirnames.append(d)

                for filename in filenames:
                    rel_filepath = os.path.join(rel_dirpath, filename)
                    full_path = os.path.join(dirpath, filename)
                    if filename.startswith('.'): skipped_info.append(f"Hidden File: '{rel_filepath}'"); continue
                    if os.path.splitext(filename)[1].lower() not in allowed_extensions_lower: skipped_info.append(
                        f"Wrong Ext: '{rel_filepath}'"); continue
                    try:
                        if not os.access(full_path, os.R_OK): skipped_info.append(
                            f"Unreadable: '{rel_filepath}'"); continue
                        size = os.path.getsize(full_path)
                        if size == 0: skipped_info.append(f"Empty: '{rel_filepath}'"); continue
                        if size > max_size_bytes: skipped_info.append(
                            f"Too Large ({size / (1024 * 1024):.1f}MB): '{rel_filepath}'"); continue
                    except OSError as e:
                        skipped_info.append(f"OS Error ('{rel_filepath}'): {e.strerror}")
                        continue
                    valid_files.append(full_path)
        except OSError as e:
            skipped_info.append(f"OS Walk Error ('{root_dir}'): {e.strerror}")
        except Exception as e:
            skipped_info.append(f"Unexpected Scan Error ('{root_dir}'): {type(e).__name__}")
        logger.info(
            f"Scan of '{os.path.basename(root_dir)}' found {len(valid_files)} files. Skipped/Errors: {len(skipped_info)}")
        return valid_files, skipped_info