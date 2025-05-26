# services/upload_service.py
import datetime
import logging
import os
from html import escape
from typing import List, Tuple, Optional, Set, Dict, Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    SentenceTransformer = None  # type: ignore
    EMBEDDINGS_AVAILABLE = False
    logging.error(f"UploadService: SentenceTransformer library failed ({e}). RAG will fail.")

try:
    import numpy  # type: ignore

    NUMPY_AVAILABLE = True
except ImportError:
    numpy = None  # type: ignore
    NUMPY_AVAILABLE = False
    logging.error("UploadService: Numpy library not found. RAG DB cannot function.")

from utils import constants
from core.models import SYSTEM_ROLE, ChatMessage, ERROR_ROLE

try:
    from services.chunking_service import ChunkingService

    CHUNKING_SERVICE_AVAILABLE = True
except ImportError as e:
    ChunkingService = None  # type: ignore
    CHUNKING_SERVICE_AVAILABLE = False
    logging.error(
        f"UploadService: Failed to import ChunkingService ({e}).")
try:
    from services.vector_db_service import VectorDBService, GLOBAL_COLLECTION_ID

    VECTOR_DB_SERVICE_AVAILABLE = True
except ImportError as e:
    VectorDBService = None  # type: ignore
    VECTOR_DB_SERVICE_AVAILABLE = False
    logging.error(
        f"UploadService: Failed to import VectorDBService ({e}).")
    GLOBAL_COLLECTION_ID = "global_knowledge_fallback"  # Fallback if import fails
try:
    from services.file_handler_service import FileHandlerService

    FILE_HANDLER_SERVICE_AVAILABLE = True
except ImportError as e:
    FileHandlerService = None  # type: ignore
    FILE_HANDLER_SERVICE_AVAILABLE = False
    logging.error(
        f"UploadService: Failed to import FileHandlerService ({e}).")
try:
    from services.code_analysis_service import CodeAnalysisService

    CODE_ANALYSIS_SERVICE_AVAILABLE = True
except ImportError as e:
    CodeAnalysisService = None  # type: ignore
    CODE_ANALYSIS_SERVICE_AVAILABLE = False
    logging.error(
        f"UploadService: Failed to import CodeAnalysisService ({e}).")

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

        if not all([EMBEDDINGS_AVAILABLE, CHUNKING_SERVICE_AVAILABLE, VECTOR_DB_SERVICE_AVAILABLE,
                    FILE_HANDLER_SERVICE_AVAILABLE, CODE_ANALYSIS_SERVICE_AVAILABLE, NUMPY_AVAILABLE]):
            logger.critical("UploadService cannot initialize due to missing critical dependencies.")
            return

        try:
            self._embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            if self._embedder:  # Check if embedder loaded
                test_embedding = self._embedder.encode(["test"])
                if test_embedding is not None and hasattr(test_embedding, 'shape'):
                    self._index_dim = test_embedding.shape[1]
                else:
                    raise ValueError("Failed to get embedding shape.")
            else:
                raise ValueError("SentenceTransformer failed to initialize.")

            if self._index_dim <= 0: raise ValueError("Failed to determine embedding dimension.")
            logger.info(f"Detected embedding dimension: {self._index_dim}")

            self._file_handler_service = FileHandlerService()
            self._chunking_service = ChunkingService(
                chunk_size=getattr(constants, 'RAG_CHUNK_SIZE', 1000),
                chunk_overlap=getattr(constants, 'RAG_CHUNK_OVERLAP', 150)
            )
            self._vector_db_service = VectorDBService(index_dimension=self._index_dim)
            self._code_analysis_service = CodeAnalysisService()
            # VectorDBService.is_ready() without args checks client, not specific collection.
            self._dependencies_ready = self._vector_db_service.is_ready()
            if self._dependencies_ready:
                logger.info("UploadService initialized successfully.")
            else:
                logger.error("UploadService init: VectorDBService not ready (client init failed).")
        except Exception as e:
            logger.exception(f"CRITICAL FAILURE during UploadService component initialization: {e}")
            self._dependencies_ready = False

    def is_vector_db_ready(self, collection_id: Optional[str] = None) -> bool:
        if not self._dependencies_ready or not self._vector_db_service: return False
        # If no collection_id, check general client readiness.
        # If collection_id is provided, VectorDBService.is_ready(collection_id) checks for that specific collection.
        return self._vector_db_service.is_ready(collection_id)

    def _send_batch_to_db(self, collection_id: str,
                          batch_contents: List[str],
                          batch_embeddings: List[List[float]],
                          batch_metadatas: List[Dict[str, Any]],
                          files_in_this_batch_names: Set[str]) -> Tuple[bool, Set[str], int]:
        if not batch_contents: return True, set(), 0
        if not self._vector_db_service:  # Should not happen if _dependencies_ready is true
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
                return False, set(), num_embeddings_in_batch
        except Exception as e:
            logger.exception(
                f"Exception during batch add to coll '{collection_id}': {e}. Files: {files_in_this_batch_names}")
            return False, set(), num_embeddings_in_batch

    def process_files_for_context(self, file_paths: List[str], collection_id: str) -> Optional[
        ChatMessage]:  # type: ignore
        # MODIFICATION: collection_id is now a mandatory parameter.
        if not collection_id:
            logger.error("UploadService: process_files_for_context called without a collection_id.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: Collection ID for RAG processing is missing.]"])  # type: ignore

        if not isinstance(file_paths, list):
            return ChatMessage(role=ERROR_ROLE, parts=["[System Error: Invalid input provided.]"])  # type: ignore

        num_input_files = len(file_paths)
        logger.info(f"Processing {num_input_files} files for RAG collection '{collection_id}'...")

        overall_successfully_added_files = set()
        processing_error_files_dict = {}
        db_failed_files_exclusive_set = set()
        binary_skipped_files = set()
        no_content_files = set()

        total_chunks_generated = 0
        total_embeddings_in_successful_batches = 0
        total_embeddings_in_failed_batches = 0
        any_db_batch_add_failure_occurred = False

        if not self.is_vector_db_ready(collection_id):  # Check readiness for the target collection
            return ChatMessage(role=ERROR_ROLE,
                               parts=[f"[Error: RAG DB/collection '{collection_id}' not ready.]"])  # type: ignore

        if not self._embedder or not self._file_handler_service or not self._chunking_service:
            logger.error("UploadService: Core components (embedder, file_handler, chunking_service) not initialized.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: RAG processing components not ready.]"])  # type: ignore

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
                except Exception:
                    processing_error_files_dict[escape(display_name)] = "Code Parse Error"

            if escape(display_name) in processing_error_files_dict: continue
            if not content:  # If content is None or empty string after successful read
                no_content_files.add(escape(display_name));
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
                                    # type: ignore
                                    'start_line', -1) and s.get("name")]
                    md['code_entities'] = ", ".join(sorted(list(set(entities)))) if entities else ""
                    md['collection_id'] = collection_id  # Ensure metadata includes the target collection
                    file_specific_chunk_contents.append(chunk_data['content'])
                    file_specific_metadatas.append(md)

                if not file_specific_chunk_contents: no_content_files.add(escape(display_name)); continue

                embeddings_np = self._embedder.encode(file_specific_chunk_contents, show_progress_bar=False)

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

        if num_input_files == 0: return ChatMessage(role=SYSTEM_ROLE,  # type: ignore
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

        # MODIFICATION: Clarify which collection_id the summary is for
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
                           # type: ignore
                           metadata={
                               "upload_summary_v5": {  # Renamed for clarity
                                   "input_files": num_input_files,
                                   "successfully_added_files": len(overall_successfully_added_files),
                                   "embeddings_in_successful_batches": total_embeddings_in_successful_batches,
                                   "processing_error_files": len(processing_error_files_dict),
                                   "db_failed_files_exclusive": len(db_failed_files_exclusive_set),
                                   "embeddings_in_failed_db_batches": total_embeddings_in_failed_batches,
                                   "binary_skipped_files": len(binary_skipped_files),
                                   "no_content_files": len(actual_no_content_files),
                                   "collection_id": collection_id,  # MODIFICATION: Include collection_id in metadata
                               }, "total_chunks_generated": total_chunks_generated
                           })

    def process_directory_for_context(self, dir_path: str, collection_id: str) -> Optional[ChatMessage]:  # type: ignore
        # MODIFICATION: collection_id is now a mandatory parameter.
        logger.info(f"Processing directory '{dir_path}' for RAG collection '{collection_id}'")
        if not collection_id:
            logger.error("UploadService: process_directory_for_context called without a collection_id.")
            return ChatMessage(role=ERROR_ROLE,
                               parts=["[System Error: Collection ID for RAG processing is missing.]"])  # type: ignore
        try:
            if not os.path.isdir(dir_path): return ChatMessage(role=ERROR_ROLE, parts=[  # type: ignore
                f"[Error: Not a directory: '{os.path.basename(dir_path)}']"])

            if not self._file_handler_service:
                logger.error("UploadService: FileHandlerService not initialized in process_directory_for_context.")
                return ChatMessage(role=ERROR_ROLE,
                                   parts=["[System Error: File handling service not ready.]"])  # type: ignore

            valid_files, skipped_scan_info = self._scan_directory(dir_path)
            if not valid_files:
                msg = f"[RAG Scan: No suitable files in '{os.path.basename(dir_path)}' for collection '{collection_id}'."
                if skipped_scan_info:
                    msg += f" {len(skipped_scan_info)} items skipped/errored during scan.]"
                else:
                    msg += "]"
                return ChatMessage(role=SYSTEM_ROLE, parts=[msg])  # type: ignore

            # MODIFICATION: Pass the collection_id to process_files_for_context
            process_msg_obj = self.process_files_for_context(valid_files, collection_id=collection_id)
            if process_msg_obj and skipped_scan_info:
                scan_issue_txt = f"{len(skipped_scan_info)} items skipped/error in dir scan."
                if process_msg_obj.metadata: process_msg_obj.metadata[
                    "dir_scan_issues"] = scan_issue_txt  # type: ignore
                if process_msg_obj.parts and isinstance(process_msg_obj.parts[0], str):  # type: ignore
                    process_msg_obj.parts[0] = process_msg_obj.parts[0].rstrip(
                        ' .]') + f" | DirScan: {scan_issue_txt}]"  # type: ignore
            return process_msg_obj
        except Exception as e:
            logger.exception(f"CRITICAL ERROR processing directory '{dir_path}' for collection '{collection_id}': {e}")
            return ChatMessage(role=ERROR_ROLE,  # type: ignore
                               parts=[
                                   f"[System: Critical error processing directory '{os.path.basename(dir_path)}' for collection '{collection_id}'.]"])

    def query_vector_db(self, query_text: str, collection_ids: List[str],
                        n_results: int = constants.RAG_NUM_RESULTS) -> List[Dict[str, Any]]:  # type: ignore
        # MODIFICATION: collection_ids is now a list and can contain multiple collection names
        if not self._dependencies_ready or not self._embedder or not self._vector_db_service: return []
        if not query_text.strip(): return []
        n_results = max(1, n_results)  # type: ignore
        if not collection_ids:
            logger.warning(
                "UploadService.query_vector_db called with empty collection_ids, defaulting to GLOBAL_COLLECTION_ID.")
            collection_ids = [GLOBAL_COLLECTION_ID]

        all_results: List[Dict[str, Any]] = []
        try:
            query_embedding = self._embedder.encode([query_text]).tolist()
            if not query_embedding or not query_embedding[0] or len(query_embedding[0]) != self._index_dim:
                logger.error(
                    f"Failed to generate valid query embedding or dimension mismatch. Expected {self._index_dim}")
                return []

            for coll_id in collection_ids:
                if not coll_id or not isinstance(coll_id, str):
                    logger.warning(f"Skipping invalid collection_id in query: {coll_id}")
                    continue
                if self._vector_db_service.is_ready(coll_id):
                    logger.debug(f"Querying collection: {coll_id} for '{query_text[:30]}...'")
                    coll_results = self._vector_db_service.search(coll_id, query_embedding, k=n_results)  # type: ignore
                    if coll_results:
                        for res_item in coll_results:
                            if 'metadata' not in res_item or res_item['metadata'] is None:
                                res_item['metadata'] = {}
                            res_item['metadata']['retrieved_from_collection'] = coll_id  # Add source collection
                        all_results.extend(coll_results)
                else:
                    logger.warning(f"Collection '{coll_id}' not ready, skipping in query.")
            return all_results
        except Exception as e:
            logger.exception(f"Error querying Vector DB: {e}")
            return []

    def _scan_directory(self, root_dir: str, allowed_extensions: Set[str] = constants.ALLOWED_TEXT_EXTENSIONS,
                        # type: ignore
                        ignored_dirs: Set[str] = constants.DEFAULT_IGNORED_DIRS) -> Tuple[
        List[str], List[str]]:  # type: ignore
        valid_files, skipped_info = [], []
        logger.info(f"Scanning directory: {root_dir}")
        ignored_dirs_lower = {d.lower() for d in ignored_dirs if isinstance(d, str)}  # type: ignore
        allowed_extensions_lower = {e.lower() for e in allowed_extensions if isinstance(e, str)}  # type: ignore
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
                    depth = rel_dirpath.count(os.sep) if rel_dirpath != '.' else 0
                except ValueError:
                    logger.warning(f"Cannot get relative path for {dirpath} from {root_dir}. Assuming depth 0.")
                    depth = 0
                    rel_dirpath = dirpath

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

