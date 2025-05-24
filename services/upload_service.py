# services/upload_service.py
# UPDATED: Added logging after batch add to VectorDB
# MODIFIED: Adapted for ChromaDB (passing content strings)
# FIXED: Changed relative imports to absolute imports for service dependencies

import datetime
import logging
import os
from html import escape
from typing import List, Tuple, Optional, Set, Dict, Any

import numpy as np  # Still needed for SentenceTransformer output

# --- Dependency Imports ---
try:
    from sentence_transformers import SentenceTransformer

    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using your current default, will evaluate alternatives later
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    SentenceTransformer = None  # type: ignore
    EMBEDDINGS_AVAILABLE = False
    logging.error(f"UploadService: SentenceTransformer library failed ({e}). RAG will fail.")

try:
    import numpy  # Needed for embedding array operations

    NUMPY_AVAILABLE = True
except ImportError:
    numpy = None  # type: ignore
    NUMPY_AVAILABLE = False
    logging.error("UploadService: Numpy library not found. RAG DB cannot function.")

# --- Local Imports ---
from utils import constants
from core.models import SYSTEM_ROLE, ChatMessage, ERROR_ROLE

# Import ChunkingService
try:
    # FIXED: Changed from 'from .chunking_service' to 'from services.chunking_service'
    from services.chunking_service import ChunkingService

    CHUNKING_SERVICE_AVAILABLE = True
except ImportError as e:
    ChunkingService = None  # type: ignore
    CHUNKING_SERVICE_AVAILABLE = False
    logging.error(f"UploadService: Failed to import ChunkingService ({e}). Chunking will fail.")
# Import VectorDBService
try:
    # FIXED: Changed from 'from .vector_db_service' to 'from services.vector_db_service'
    from services.vector_db_service import VectorDBService  # This now points to ChromaDB implementation

    VECTOR_DB_SERVICE_AVAILABLE = True
except ImportError as e:
    VectorDBService = None  # type: ignore
    VECTOR_DB_SERVICE_AVAILABLE = False
    logging.error(f"UploadService: Failed to import VectorDBService ({e}). RAG DB cannot function.")
# Import FileHandlerService
try:
    # FIXED: Changed from 'from .file_handler_service' to 'from services.file_handler_service'
    from services.file_handler_service import FileHandlerService

    FILE_HANDLER_SERVICE_AVAILABLE = True
except ImportError as e:
    FileHandlerService = None  # type: ignore
    FILE_HANDLER_SERVICE_AVAILABLE = False
    logging.error(f"UploadService: Failed to import FileHandlerService ({e}). File reading will fail.")
# Import CodeAnalysisService
try:
    # FIXED: Changed from 'from .code_analysis_service' to 'from services.code_analysis_service'
    from services.code_analysis_service import CodeAnalysisService

    CODE_ANALYSIS_SERVICE_AVAILABLE = True
except ImportError as e:
    CodeAnalysisService = None  # type: ignore
    CODE_ANALYSIS_SERVICE_AVAILABLE = False
    logging.error(f"UploadService: Failed to import CodeAnalysisService ({e}). Code parsing will fail.")

logger = logging.getLogger(__name__)


class UploadService:
    """
    Handles processing of uploaded files/directories for RAG. Orchestrates reading,
    chunking, embedding, and adding/querying the vector store, using CodeAnalysisService
    for structure extraction.
    """

    def __init__(self):
        logger.info("UploadService initializing...")
        self._embedder: Optional[SentenceTransformer] = None
        self._chunking_service: Optional[ChunkingService] = None
        self._vector_db_service: Optional[VectorDBService] = None
        self._file_handler_service: Optional[FileHandlerService] = None
        self._code_analysis_service: Optional[CodeAnalysisService] = None
        self._index_dim = -1
        self._dependencies_ready = False

        if not EMBEDDINGS_AVAILABLE: logger.critical(
            "UploadService cannot initialize: SentenceTransformer failed."); return
        if not CHUNKING_SERVICE_AVAILABLE: logger.critical(  # This will now reflect successful import or real error
            "UploadService cannot initialize: ChunkingService failed."); return
        if not VECTOR_DB_SERVICE_AVAILABLE: logger.critical(  # This will now reflect successful import or real error
            "UploadService cannot initialize: VectorDBService failed."); return
        if not FILE_HANDLER_SERVICE_AVAILABLE: logger.critical(  # This will now reflect successful import or real error
            "UploadService cannot initialize: FileHandlerService failed."); return
        if not CODE_ANALYSIS_SERVICE_AVAILABLE: logger.critical(
            # This will now reflect successful import or real error
            "UploadService cannot initialize: CodeAnalysisService failed."); return
        if not NUMPY_AVAILABLE: logger.critical("UploadService cannot initialize: Numpy failed."); return

        try:
            logger.info(f"Initializing embedder: {DEFAULT_EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            dummy_emb = self._embedder.encode(["test"])
            self._index_dim = dummy_emb.shape[1]
            if self._index_dim <= 0: raise ValueError("Failed to determine embedding dimension.")
            logger.info(f"Detected embedding dimension: {self._index_dim}")

            logger.info("Initializing FileHandlerService...")
            # Instantiate the imported class
            self._file_handler_service = FileHandlerService()  # type: ignore

            logger.info("Initializing ChunkingService...")
            rag_chunk_size = getattr(constants, 'RAG_CHUNK_SIZE', 1000)
            rag_chunk_overlap = getattr(constants, 'RAG_CHUNK_OVERLAP', 150)
            logger.info(
                f"  Using RAG_CHUNK_SIZE={rag_chunk_size}, RAG_CHUNK_OVERLAP={rag_chunk_overlap} from constants.")
            # Instantiate the imported class
            self._chunking_service = ChunkingService(  # type: ignore
                chunk_size=rag_chunk_size,
                chunk_overlap=rag_chunk_overlap
            )

            logger.info("Initializing VectorDBService (ChromaDB)...")
            # Instantiate the imported class
            self._vector_db_service = VectorDBService(index_dimension=self._index_dim)  # type: ignore

            logger.info("Initializing CodeAnalysisService...")
            # Instantiate the imported class
            self._code_analysis_service = CodeAnalysisService()  # type: ignore

            # Ensure VectorDBService is ready before marking UploadService ready
            self._dependencies_ready = (
                    self._embedder is not None and
                    self._file_handler_service is not None and
                    self._chunking_service is not None and
                    self._vector_db_service is not None and
                    self._code_analysis_service is not None and
                    NUMPY_AVAILABLE and
                    self._vector_db_service.is_ready()  # Check if Chroma client initialized and global collection ready
            )

            if self._dependencies_ready:
                logger.info("UploadService initialized successfully with all components.")
            else:
                logger.error(
                    "UploadService initialized BUT one or more components failed (check logs). RAG may not function.")

        except Exception as e:
            logger.exception(f"CRITICAL FAILURE during UploadService component initialization: {e}")
            self._dependencies_ready = False

    def is_vector_db_ready(self, collection_id: Optional[str] = None) -> bool:
        """
        Checks if the UploadService and its underlying vector database are ready.

        Args:
            collection_id (Optional[str]): If provided, checks if that specific collection is ready.

        Returns:
            bool: True if ready, False otherwise.
        """
        base_ready = (
                self._embedder is not None and
                self._file_handler_service is not None and
                self._chunking_service is not None and
                self._code_analysis_service is not None and
                self._vector_db_service is not None and
                NUMPY_AVAILABLE
        )
        if not base_ready:
            return False
        # The VectorDBService.is_ready(collection_id) now handles the ChromaDB check
        return self._vector_db_service.is_ready(collection_id)

    def process_files_for_context(self, file_paths: List[str], collection_id: str = constants.GLOBAL_COLLECTION_ID) -> \
            Optional[ChatMessage]:
        """
        Processes a list of file paths by reading, chunking, embedding, and adding them
        to the specified vector database collection.

        Args:
            file_paths (List[str]): Absolute paths to the files to process.
            collection_id (str): The ID of the collection to add the processed chunks to.

        Returns:
            Optional[ChatMessage]: A ChatMessage summarizing the upload process, or None on critical error.
        """
        if not isinstance(file_paths, list):
            logger.error(f"Invalid file_paths argument: Expected list, got {type(file_paths)}")
            return ChatMessage(role=ERROR_ROLE, parts=["[System Error: Invalid input provided.]"])

        num_files = len(file_paths)
        logger.info(f"UploadService: Processing {num_files} files for RAG DB collection '{collection_id}'...")
        processed_files_display = []
        binary_files = []
        error_files = []
        files_processed_for_db = 0
        chunks_generated_total = 0
        files_added_successfully_set = set()
        db_add_errors_occurred = False

        if not self.is_vector_db_ready(collection_id):  # Use collection_id specific readiness
            logger.error(f"RAG DB components not ready or collection '{collection_id}' could not be accessed/created.")
            return ChatMessage(role=ERROR_ROLE, parts=[
                f"[Error: RAG components not ready or collection '{collection_id}' inaccessible.]"],
                               metadata={"upload_error": f"Collection '{collection_id}' inaccessible"})

        # New lists to hold data formatted for ChromaDB's add method
        all_chunk_contents = []  # List of strings for Chroma's 'documents'
        all_embeddings = []  # List of lists of floats for Chroma's 'embeddings'
        all_metadatas = []  # List of dicts for Chroma's 'metadatas'
        files_in_batch = []  # Keep track of which files contributed to this batch

        for i, file_path in enumerate(file_paths):
            display_name = os.path.basename(file_path)
            if not os.path.exists(file_path): error_files.append(escape(display_name) + " (Not Found)"); logger.warning(
                f"File not found: {file_path}"); continue
            if not os.path.isfile(file_path): error_files.append(
                escape(display_name) + " (Not a File)"); logger.warning(f"Path is not a file: {file_path}"); continue
            logger.info(f"  Processing [{i + 1}/{num_files}]: {display_name}")
            try:
                read_result = self._file_handler_service.read_file_content(file_path)  # type: ignore
            except Exception as e_read:
                logger.exception(f"  Unexpected error calling FileHandlerService for '{display_name}': {e_read}")
                error_files.append(escape(display_name) + " (Read Service Error)")
                continue
            if read_result is None or not isinstance(read_result, tuple) or len(read_result) != 3:
                logger.error(f"  Invalid result from FileHandlerService reading '{display_name}'. Skipping file.")
                error_files.append(escape(display_name) + " (Internal Read Error)")
                continue
            content, file_type, error_msg = read_result
            if file_type == "error": error_files.append(
                escape(display_name) + f" ({error_msg or 'Read Error'})"); continue
            if file_type == "binary": binary_files.append(escape(display_name)); logger.info(
                f"  Skipping binary file: {display_name}"); continue
            code_structures: List[Dict[str, Any]] = []
            if file_type == "text" and content is not None:
                files_processed_for_db += 1
                processed_files_display.append(escape(display_name))
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == '.py' and self._code_analysis_service:
                    try:
                        code_structures = self._code_analysis_service.parse_python_structures(content,
                                                                                              file_path)  # type: ignore
                    except Exception as e_parse:
                        logger.error(f"Error calling CodeAnalysisService for '{display_name}': {e_parse}",
                                     exc_info=True)
                        if escape(display_name) + " (Read Error)" not in error_files:
                            error_files.append(escape(display_name) + " (Code Parse Error)")
                elif file_ext == '.py' and not self._code_analysis_service:
                    logger.warning(f"Cannot parse Python file '{display_name}', CodeAnalysisService not available.")
                chunks = []
                # `chunk_contents_for_embedding` is now the primary content list for Chroma
                chunk_contents_for_embedding = []
                enhanced_metadata_list_for_file = []  # This is for Chroma's 'metadatas'
                try:
                    logger.debug(f"  Calling ChunkingService for '{display_name}' (ext: {file_ext})")
                    chunks = self._chunking_service.chunk_document(content, source_id=file_path,
                                                                   file_ext=file_ext)  # type: ignore
                    if not chunks:
                        logger.warning(f"  No chunks generated by ChunkingService for '{display_name}'.")
                        if content.strip():
                            if escape(display_name) + " (Read Error)" not in error_files:
                                error_files.append(escape(display_name) + " (No Chunks)")
                        continue
                    logger.info(f"  Generated {len(chunks)} chunks for '{display_name}'.")
                    chunks_generated_total += len(chunks)
                    logger.debug(
                        f"Enhancing metadata and preparing embeddings for {len(chunks)} chunks from '{display_name}'...")
                    for chunk_idx, chunk_data in enumerate(chunks):
                        if not isinstance(chunk_data,
                                          dict) or 'metadata' not in chunk_data or 'content' not in chunk_data:
                            logger.warning(f"Skipping invalid chunk data at index {chunk_idx} in '{display_name}'.")
                            continue
                        chunk_metadata = chunk_data['metadata']
                        chunk_content = chunk_data['content']
                        chunk_start_line = chunk_metadata.get('start_line')
                        chunk_end_line = chunk_metadata.get('end_line')
                        overlapping_entities = []
                        if code_structures and chunk_start_line is not None and chunk_end_line is not None:
                            for struct in code_structures:
                                struct_start = struct.get("start_line")
                                struct_end = struct.get("end_line")
                                struct_name = struct.get("name")
                                if struct_start is not None and struct_end is not None and struct_name:
                                    if struct_start <= chunk_end_line and struct_end >= chunk_start_line:
                                        overlapping_entities.append(struct_name)

                        # Add code_entities to metadata for ChromaDB
                        chunk_metadata['code_entities'] = overlapping_entities
                        if overlapping_entities:
                            logger.debug(
                                f"Chunk {chunk_idx} ({chunk_start_line}-{chunk_end_line}) in '{display_name}' associated with: {overlapping_entities}")

                        # Add collection_id to metadata for ChromaDB
                        chunk_metadata['collection_id'] = collection_id

                        chunk_contents_for_embedding.append(chunk_content)
                        enhanced_metadata_list_for_file.append(chunk_metadata)

                    if not chunk_contents_for_embedding:
                        logger.warning(f"No valid chunk content to embed for '{display_name}'.")
                        continue

                    logger.debug(f"  Encoding {len(chunk_contents_for_embedding)} chunks for '{display_name}'...")
                    embeddings_np = self._embedder.encode(chunk_contents_for_embedding,
                                                          show_progress_bar=False)  # type: ignore

                    # Convert numpy array to list of lists of floats for ChromaDB
                    embeddings_list_of_lists = embeddings_np.tolist()

                    if len(embeddings_list_of_lists) == len(enhanced_metadata_list_for_file) and \
                            len(embeddings_list_of_lists) == len(
                        chunk_contents_for_embedding):  # All lengths must match
                        all_chunk_contents.extend(chunk_contents_for_embedding)
                        all_embeddings.extend(embeddings_list_of_lists)
                        all_metadatas.extend(enhanced_metadata_list_for_file)
                        files_in_batch.append(display_name)
                    else:
                        logger.error(
                            f"Mismatch between embeddings ({len(embeddings_list_of_lists)}), contents ({len(chunk_contents_for_embedding)}) and enhanced metadata ({len(enhanced_metadata_list_for_file)}) for '{display_name}'. Skipping DB add for this file.")
                        if escape(display_name) + " (Processing Error)" not in error_files:
                            error_files.append(escape(display_name) + " (Metadata/Embedding Error)")
                except Exception as e_proc:
                    logger.exception(
                        f"  Error during chunking, metadata enhancement, or embedding for {display_name}: {e_proc}")
                    if escape(display_name) + " (Read Error)" not in error_files and escape(
                            display_name) + " (Code Parse Error)" not in error_files:
                        error_files.append(escape(display_name) + " (Processing Error)")
                    continue
            else:
                logger.warning(f"  Skipping file '{display_name}' due to unexpected type/content state ({file_type}).")

        batch_add_success = False
        num_embeddings_in_batch = len(all_embeddings)  # Total count for the batch
        if all_embeddings:  # If there are any embeddings to add
            try:
                logger.info(
                    f"Adding batch of {num_embeddings_in_batch} documents from {len(files_in_batch)} files to collection '{collection_id}'...")

                # Pass all three lists to VectorDBService.add_embeddings
                batch_add_success = self._vector_db_service.add_embeddings(  # type: ignore
                    collection_id,
                    all_chunk_contents,  # Contents (documents)
                    all_embeddings,  # Embeddings (vectors)
                    all_metadatas  # Metadatas
                )

                if batch_add_success:
                    logger.debug(
                        f"[RAG DEBUG] Successfully added {num_embeddings_in_batch} documents to collection '{collection_id}'. Files: {files_in_batch}")
                    files_added_successfully_set.update(files_in_batch)
                else:
                    logger.error(f"[RAG DEBUG] Failed to add batch to collection '{collection_id}'.")
                    db_add_errors_occurred = True
                    for f_name in files_in_batch:
                        if f_name not in [err.split(" (")[0] for err in error_files]:
                            error_files.append(escape(f_name) + " (DB Batch Add Failed)")
            except Exception as e_batch_add:
                logger.exception(
                    f"Critical error during batch add to '{collection_id}': {e_batch_add}")
                db_add_errors_occurred = True
                for f_name in files_in_batch:
                    if f_name not in [err.split(" (")[0] for err in error_files]:
                        error_files.append(escape(f_name) + " (DB Batch Add Error)")
        else:
            logger.info(f"No documents were generated or accumulated to add to collection '{collection_id}'.")

        if not processed_files_display and not binary_files and not error_files:
            return ChatMessage(role=SYSTEM_ROLE,
                               parts=[f"[Upload Info: No files provided to process for collection '{collection_id}'.]"])

        status_notes = []
        num_processed_text_files = len(processed_files_display)
        num_unique_error_files = len(set([err.split(" (")[0] for err in error_files if " (" in err])) + len(
            set([err for err in error_files if " (" not in err]))
        num_binary = len(binary_files)
        num_actually_added_files = len(files_added_successfully_set)

        if num_actually_added_files > 0:
            status_notes.append(
                f"{num_actually_added_files} file(s) added ({num_embeddings_in_batch} embeddings)")
        elif num_processed_text_files > 0 and not db_add_errors_occurred:
            status_notes.append(f"{num_processed_text_files} text file(s) processed, but none added to DB (check logs)")
        elif db_add_errors_occurred:
            status_notes.append("DB add errors occurred (check logs)")

        if num_unique_error_files > 0: status_notes.append(f"{num_unique_error_files} file(s) with errors")
        if num_binary > 0: status_notes.append(f"{num_binary} binary file(s) ignored")

        problem_files_display = []
        if error_files: problem_files_display.extend(error_files)
        if binary_files: problem_files_display.extend([f"{f} (Binary)" for f in binary_files])

        # --- MODIFIED MESSAGE ROLE ---
        message_role = SYSTEM_ROLE  # Default to SYSTEM_ROLE
        if num_unique_error_files > 0 or db_add_errors_occurred:
            message_role = ERROR_ROLE
        # --- END MODIFIED MESSAGE ROLE ---

        summary_parts = []
        summary_parts.append(f"Upload Processed {num_files} item(s) for collection '{collection_id}'.")
        if status_notes: summary_parts.append("Status: " + "; ".join(status_notes) + ".")
        if problem_files_display:
            problem_list_str = ", ".join(f"'{f}'" for f in problem_files_display)
            if len(problem_list_str) > 200: problem_list_str = problem_list_str[:197] + "...'"
            summary_parts.append(f"Issues with: {problem_list_str}")

        summary_text = "[RAG " + " ".join(summary_parts) + "]"
        upload_timestamp = datetime.datetime.now().isoformat()
        logger.info(f"UploadService: Upload processing finished for '{collection_id}'. Summary: {summary_text}")
        return ChatMessage(role=message_role, parts=[summary_text], timestamp=upload_timestamp,
                           metadata={
                               "upload_summary": f"{num_actually_added_files}/{files_processed_for_db} processed to DB for '{collection_id}'",
                               "errors_count": num_unique_error_files, "binary_skipped": num_binary,
                               "chunks_generated": chunks_generated_total, "collection_id": collection_id})

    def process_directory_for_context(self, dir_path: str, collection_id: str = constants.GLOBAL_COLLECTION_ID) -> \
            Optional[ChatMessage]:
        """
        Scans a directory for relevant files, processes them, and adds them to the RAG DB.

        Args:
            dir_path (str): The absolute path to the directory to scan.
            collection_id (str): The ID of the collection to add the processed chunks to.

        Returns:
            Optional[ChatMessage]: A ChatMessage summarizing the directory processing, or None on critical error.
        """
        logger.info(f"UploadService: Processing directory '{dir_path}' for RAG DB collection '{collection_id}'")
        try:
            if not isinstance(dir_path, str) or not dir_path:
                return ChatMessage(role=ERROR_ROLE, parts=["[Error: Invalid directory path provided.]"])
            if not os.path.isdir(dir_path):
                logger.error(f"Path is not a directory: {dir_path}")
                return ChatMessage(role=ERROR_ROLE, parts=[f"[Error: Not a directory: '{os.path.basename(dir_path)}']"])
            valid_files, skipped_info = self._scan_directory(dir_path)
            if not valid_files:
                logger.warning(f"Scan of '{os.path.basename(dir_path)}' found no allowed/readable files.")
                scan_summary_msg = f"[RAG Upload Scan: Scan of '{os.path.basename(dir_path)}' for collection '{collection_id}' found no suitable files."
                if skipped_info: scan_summary_msg += f" Skipped/Errors during scan: {len(skipped_info)} (see logs)."
                scan_summary_msg += "]"
                return ChatMessage(role=SYSTEM_ROLE, parts=[scan_summary_msg],
                                   metadata={"collection_id": collection_id, "scan_skipped_count": len(skipped_info)})
            logger.info(
                f"Directory scan found {len(valid_files)} files. Processing for RAG DB collection '{collection_id}'...")
            context_message = self.process_files_for_context(valid_files, collection_id=collection_id)
            if context_message and skipped_info:
                scan_summary_detail = f"{len(skipped_info)} items skipped/error during scan (see logs)."
                if context_message.metadata:
                    context_message.metadata["scan_summary"] = scan_summary_detail
                else:
                    context_message.metadata = {"scan_summary": scan_summary_detail}
                if context_message.parts and isinstance(context_message.parts[0], str):
                    original_text = context_message.parts[0]
                    if original_text.strip().endswith(']'):
                        summary_text = original_text.rstrip()[:-1] + f" | Scan Issues: {scan_summary_detail}]"
                    else:
                        summary_text = original_text + f" [Scan Issues: {scan_summary_detail}]"
                    context_message.parts[0] = summary_text
                    logger.debug(f"Appended scan summary to context message text: {scan_summary_detail}")
                else:
                    logger.warning("Could not append scan summary to context message text: No text part found.")
            return context_message
        except Exception as e:
            logger.exception(
                f"CRITICAL ERROR during directory processing '{dir_path}' for collection '{collection_id}': {e}")
            return ChatMessage(role=ERROR_ROLE, parts=[
                f"[System: Critical error processing directory '{os.path.basename(dir_path)}' for collection '{collection_id}'. See logs.]"],
                               metadata={"collection_id": collection_id})

    def query_vector_db(self, query_text: str, collection_ids: List[str] = [constants.GLOBAL_COLLECTION_ID],
                        n_results: int = constants.RAG_NUM_RESULTS) -> List[Dict[str, Any]]:
        """
        Queries the vector database for relevant chunks based on a query string.

        Args:
            query_text (str): The user's query string.
            collection_ids (List[str]): List of collection IDs to query.
            n_results (int): The maximum number of results to return from each collection.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing 'content', 'metadata', and 'distance' for each relevant chunk.
        """
        if not self._dependencies_ready:
            logger.error("Cannot query Vector DB: UploadService dependencies not ready.")
            return []
        if not isinstance(query_text, str) or not query_text.strip():
            logger.warning("Attempted query Vector DB with empty or invalid text.")
            return []
        if not isinstance(n_results, int) or n_results <= 0:
            logger.warning(f"Invalid n_results ({n_results}), using default: {constants.RAG_NUM_RESULTS}")
            n_results = constants.RAG_NUM_RESULTS
        if not isinstance(collection_ids, list) or not collection_ids:
            logger.warning("No collection_ids provided for query, using default global collection.")
            collection_ids = [constants.GLOBAL_COLLECTION_ID]
        logger.info(
            f"Querying Vector DB (k={n_results}) across collections: {collection_ids} for '{query_text[:50]}...'")
        all_results: List[Dict[str, Any]] = []
        queried_collections = []
        try:
            logger.debug("Encoding query text...")
            # Encode query and convert to list of lists of floats as expected by VectorDBService.search
            query_embedding = self._embedder.encode([query_text]).tolist()  # type: ignore

            if not isinstance(query_embedding, list) or not query_embedding or not isinstance(query_embedding[0], list):
                logger.error("Encoding failed or produced invalid format for query embedding.")
                return []
            if len(query_embedding[0]) != self._index_dim:
                logger.error(
                    f"Query embedding dimension mismatch! Expected {self._index_dim}, got {len(query_embedding[0])}. Aborting.")
                return []

            for collection_id in collection_ids:
                if self._vector_db_service.is_ready(collection_id):  # type: ignore
                    logger.debug(f"  Querying collection '{collection_id}'...")
                    collection_results = self._vector_db_service.search(collection_id, query_embedding,
                                                                        k=n_results)  # type: ignore
                    if collection_results:
                        all_results.extend(collection_results)
                        queried_collections.append(collection_id)
                        logger.debug(f"  Found {len(collection_results)} results in '{collection_id}'.")
                    else:
                        logger.debug(f"  No results found in collection '{collection_id}'.")
                else:
                    logger.warning(f"  Collection '{collection_id}' is not ready for querying. Skipping.")
            if not all_results:
                logger.info("No results found across any of the queried collections.")
                return []
            # Sort all results from all queried collections by distance (lower is better)
            sorted_results = sorted([res for res in all_results if 'distance' in res],
                                    key=lambda x: x.get('distance', float('inf')))
            # Take the top N overall results
            final_results = sorted_results[:n_results]
            logger.info(
                f"Query completed across {len(queried_collections)} collections. Returning top {len(final_results)} overall results.")
            return final_results
        except Exception as e:
            logger.exception(f"Error querying Vector DB across collections {collection_ids}: {e}")
            return []

    def _scan_directory(
            self,
            root_dir: str,
            allowed_extensions: Set[str] = constants.ALLOWED_TEXT_EXTENSIONS,
            ignored_dirs: Set[str] = constants.DEFAULT_IGNORED_DIRS,
    ) -> Tuple[List[str], List[str]]:
        """
        Recursively scans a directory for files, filtering by extension, size, and ignoring specified directories.

        Args:
            root_dir (str): The starting directory for the scan.
            allowed_extensions (Set[str]): Set of file extensions to include (e.g., {'.py', '.md'}).
            ignored_dirs (Set[str]): Set of directory names to ignore (e.g., {'.git', 'venv'}).

        Returns:
            Tuple[List[str], List[str]]: A tuple containing:
                - List of absolute paths to valid files found.
                - List of strings describing skipped files/directories.
        """
        valid_files: List[str] = []
        skipped_info: List[str] = []
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
                    if dirpath == root_dir:
                        depth = 0
                        rel_dirpath = '.'
                    else:
                        try:
                            relative_part = os.path.relpath(dirpath,
                                                            root_dir)
                            rel_dirpath = relative_part
                            depth = relative_part.count(
                                os.sep) if relative_part and relative_part != '.' else 0
                        except ValueError:
                            logger.warning(
                                f"Cannot get relative path for {dirpath} from {root_dir}. Skipping dir content.")
                            dirnames[
                            :] = []
                            continue
                except Exception as e_depth:
                    logger.warning(
                        f"Error calculating directory depth for {dirpath}: {e_depth}. Skipping dir content.")
                    dirnames[
                    :] = []
                    continue
                if depth >= max_depth: skipped_info.append(f"Max Depth ({max_depth}): '{rel_dirpath}'"); dirnames[
                                                                                                         :] = []; continue
                filtered_dirnames = []
                for d in dirnames:
                    if d.startswith('.'):
                        skipped_info.append(f"Hidden Dir: '{os.path.join(rel_dirpath, d)}'")
                    elif d.lower() in ignored_dirs_lower or d in ignored_dirs:
                        skipped_info.append(f"Ignored Dir: '{os.path.join(rel_dirpath, d)}'")
                    else:
                        filtered_dirnames.append(d)
                dirnames[:] = filtered_dirnames
                for filename in filenames:
                    rel_filepath = os.path.join(rel_dirpath, filename) if rel_dirpath != '.' else filename
                    full_path = os.path.join(dirpath, filename)
                    if filename.startswith('.'): skipped_info.append(f"Hidden File: '{rel_filepath}'"); continue
                    _, ext = os.path.splitext(filename)
                    if not ext or ext.lower() not in allowed_extensions_lower: skipped_info.append(
                        f"Wrong Ext ('{ext}'): '{rel_filepath}'"); continue
                    try:
                        if not os.access(full_path, os.R_OK): skipped_info.append(
                            f"Unreadable: '{rel_filepath}'"); continue
                        size = os.path.getsize(full_path)
                        if size == 0: skipped_info.append(f"Empty: '{rel_filepath}'"); continue
                        if size > max_size_bytes: skipped_info.append(
                            f"Too Large ({size / (1024 * 1024):.1f}MB): '{rel_filepath}'"); continue
                    except OSError as file_err:
                        skipped_info.append(f"OS Error Accessing '{rel_filepath}': {file_err}")
                        logger.warning(
                            f"OS Error accessing '{full_path}': {file_err}")
                        continue
                    except Exception as e_file:
                        skipped_info.append(f"Error Accessing '{rel_filepath}': {e_file}")
                        logger.warning(
                            f"Unexpected error checking '{full_path}': {e_file}")
                        continue
                    valid_files.append(full_path)
        except OSError as walk_err:
            error_msg = f"Error scanning directory tree ('{root_dir}'): {walk_err}"
            skipped_info.append(
                error_msg)
            logger.error(error_msg)
        except Exception as e_walk:
            error_msg = f"Unexpected error during directory scan: {e_walk}"
            skipped_info.append(
                error_msg)
            logger.exception("Directory scan failed")
        logger.info(
            f"Scan complete for '{os.path.basename(root_dir)}'. Found {len(valid_files)} valid files. Skipped/Errors: {len(skipped_info)}")
        return valid_files, skipped_info