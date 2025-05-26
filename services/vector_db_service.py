# services/vector_db_service.py
import logging
import os
import shutil
import uuid
from pathlib import Path

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None # type: ignore
    CHROMADB_AVAILABLE = False
    logging.critical(
        "VectorDBService: ChromaDB library not found. RAG DB cannot function. Install: pip install chromadb")

from typing import List, Dict, Any, Optional

from utils import constants

GLOBAL_COLLECTION_ID = constants.GLOBAL_COLLECTION_ID
logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self, index_dimension: int, base_persist_directory: Optional[str] = None):
        logger.info("VectorDBService initializing (ChromaDB implementation)...")
        self._client: Optional[chromadb.PersistentClient] = None
        self._index_dim = index_dimension

        if not CHROMADB_AVAILABLE:
            logger.critical("ChromaDB library not available. VectorDBService cannot initialize.")
            return

        if not isinstance(index_dimension, int) or index_dimension <= 0:
            logger.critical(f"Invalid index dimension provided: {index_dimension}. Cannot initialize ChromaDB.")
            return

        self.base_persist_directory = base_persist_directory or constants.RAG_COLLECTIONS_PATH
        # AVA_ASSISTANT_MODIFIED: Log the exact path being used for ChromaDB persistence
        logger.info(f"VectorDBService: ChromaDB base_persist_directory set to: {os.path.abspath(self.base_persist_directory)}")

        try:
            os.makedirs(self.base_persist_directory, exist_ok=True)
            logger.info(f"ChromaDB base directory ensured: {self.base_persist_directory}")
        except OSError as e:
            logger.critical(f"Failed to create ChromaDB base directory '{self.base_persist_directory}': {e}", exc_info=True)
            return # Critical failure if directory cannot be made

        try:
            logger.info(f"Attempting to initialize chromadb.PersistentClient with path: {os.path.abspath(self.base_persist_directory)}")
            self._client = chromadb.PersistentClient(path=self.base_persist_directory)
            logger.info("ChromaDB PersistentClient initialized successfully.")

            if not self.get_or_create_collection(GLOBAL_COLLECTION_ID):
                logger.error(
                    f"VectorDBService initialized, but the global collection ('{GLOBAL_COLLECTION_ID}') could not be properly created or loaded. "
                    "RAG functionality relying on the global context may be impaired."
                )
            else:
                logger.info("VectorDBService fully initialized and global collection is ready.")

        except Exception as e:
            logger.critical(f"CRITICAL FAILURE: Failed to initialize ChromaDB client with path '{os.path.abspath(self.base_persist_directory)}': {e}", exc_info=True)
            self._client = None

    def get_or_create_collection(self, collection_id: str) -> Optional[chromadb.Collection]:
        if not CHROMADB_AVAILABLE or self._client is None:
            logger.error(
                f"Cannot get/create collection '{collection_id}': ChromaDB not available or client not initialized.")
            return None
        if not isinstance(collection_id, str) or not collection_id.strip():
            logger.error("Cannot get/create collection: Invalid or empty collection_id provided.")
            return None
        try:
            collection = self._client.get_or_create_collection(
                name=collection_id,
                embedding_function=None
            )
            logger.debug(f"Collection '{collection_id}' accessed/created successfully.")
            return collection
        except Exception as e:
            # AVA_ASSISTANT_MODIFIED: More detailed logging for collection creation/access failure
            logger.error(f"Error getting/creating collection '{collection_id}': {type(e).__name__} - {e}", exc_info=True)
            # Attempt to list collections to see if the client is responsive
            try:
                if self._client:
                    logger.info(f"Current collections in DB: {[col.name for col in self._client.list_collections()]}")
            except Exception as list_e:
                logger.error(f"Failed to list collections after error: {list_e}")
            return None

    def is_ready(self, collection_id: Optional[str] = None) -> bool:
        if not CHROMADB_AVAILABLE or self._client is None:
            logger.debug("VectorDBService.is_ready: ChromaDB not available or client is None. Returning False.")
            return False
        if collection_id is None:
            # Check general client health by trying a benign operation
            try:
                self._client.heartbeat() # Check if client is responsive
                logger.debug("VectorDBService.is_ready: Client heartbeat successful. Returning True for general readiness.")
                return True
            except Exception as e:
                logger.error(f"VectorDBService.is_ready: Client heartbeat failed: {e}. Returning False.", exc_info=True)
                return False

        try:
            existing_collections = self._client.list_collections()
            collection_names = [coll.name for coll in existing_collections]
            logger.debug(f"VectorDBService.is_ready: Checking for collection '{collection_id}'. Existing collections: {collection_names}")
            return collection_id in collection_names
        except Exception as e:
            logger.error(f"Error checking if collection '{collection_id}' exists: {e}", exc_info=True)
            return False

    def add_embeddings(self, collection_id: str, contents: List[str], embeddings: List[List[float]],
                       metadatas: List[Dict[str, Any]]) -> bool:
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot add embeddings: Collection '{collection_id}' not found or created.")
            return False

        if not (isinstance(contents, list) and isinstance(embeddings, list) and isinstance(metadatas, list)):
            logger.error("Invalid input: contents, embeddings, and metadatas must be lists.")
            return False
        if not (len(contents) == len(embeddings) == len(metadatas)):
            logger.error(
                f"Input length mismatch: contents ({len(contents)}), embeddings ({len(embeddings)}), metadatas ({len(metadatas)}).")
            return False
        if not contents:
            logger.warning(f"No contents provided to add to collection '{collection_id}'.")
            return True

        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list) or len(emb) != self._index_dim:
                logger.error(
                    f"Embedding at index {i} has incorrect dimension or type: {len(emb) if isinstance(emb, list) else type(emb)}. Expected {self._index_dim}.")
                return False

        logger.info(f"Adding {len(contents)} documents to collection '{collection_id}'...")
        ids = [f"doc-{uuid.uuid4().hex}" for _ in range(len(contents))]

        try:
            collection.add(
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(contents)} documents to '{collection_id}'.")
            return True
        except Exception as e:
            logger.exception(f"Error adding documents to collection '{collection_id}': {e}")
            return False

    def remove_document_chunks_by_source(self, collection_id: str, source_path_to_remove: str) -> bool:
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot remove documents: Collection '{collection_id}' not found.")
            return False

        logger.info(f"Removing documents with source '{source_path_to_remove}' from '{collection_id}'...")
        try:
            collection.delete(where={"source": source_path_to_remove})
            logger.info(f"Successfully deleted documents with source '{source_path_to_remove}' from '{collection_id}'.")
            return True
        except Exception as e:
            logger.exception(f"Error removing documents by source from '{collection_id}': {e}")
            return False

    def search(self, collection_id: str, query_embedding: List[List[float]], k: int = 5) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot search: Collection '{collection_id}' not found.")
            return []

        if not isinstance(query_embedding, list) or len(query_embedding) != 1 or \
                not isinstance(query_embedding[0], list) or len(query_embedding[0]) != self._index_dim:
            logger.error(
                f"Invalid query embedding format. Expected a list containing one list of {self._index_dim} floats.")
            return []

        try:
            current_count = collection.count()
            logger.debug(f"Searching collection '{collection_id}' (count: {current_count}) with k={k}.")
            if current_count == 0:
                logger.debug(f"Collection '{collection_id}' is empty, no results to search.")
                return []
            effective_k = min(k, current_count)
            if effective_k == 0: return [] # Should be caught by count() == 0, but defensive

            query_results = collection.query(
                query_embeddings=query_embedding,
                n_results=effective_k,
                include=['documents', 'metadatas', 'distances']
            )
            results_list = []
            if query_results and query_results.get('documents') and query_results.get(
                    'metadatas') and query_results.get('distances'):
                for i in range(len(query_results['documents'][0])):
                    doc_content = query_results['documents'][0][i]
                    doc_metadata = query_results['metadatas'][0][i]
                    doc_distance = query_results['distances'][0][i]
                    results_list.append({
                        'content': doc_content,
                        'metadata': doc_metadata,
                        'distance': doc_distance
                    })
            logger.debug(f"Search in '{collection_id}' returned {len(results_list)} results.")
            return results_list
        except Exception as e:
            logger.exception(f"Error searching collection '{collection_id}': {e}")
            return []


    def get_all_metadata(self, collection_id: str) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection(collection_id)
        if collection is None: return []
        try:
            all_data = collection.get(ids=None, where=None, include=['metadatas'])
            return all_data.get('metadatas', [])
        except Exception as e:
            logger.exception(f"Error retrieving all metadata from collection '{collection_id}': {e}")
            return []

    def get_collection_size(self, collection_id: str) -> int:
        collection = self.get_or_create_collection(collection_id)
        if collection is None: return -1
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting count for collection '{collection_id}': {e}", exc_info=True)
            return -1

    def clear_collection(self, collection_id: str) -> bool:
        if collection_id == GLOBAL_COLLECTION_ID:
            logger.warning("Clearing the GLOBAL_COLLECTION is not allowed via this method for safety.")
            return False

        logger.warning(f"Clearing collection '{collection_id}' by deleting and re-creating it.")
        try:
            if self._client:
                self._client.delete_collection(name=collection_id)
                self.get_or_create_collection(collection_id)
                logger.info(f"Collection '{collection_id}' cleared and re-created successfully.")
                return True
            return False
        except Exception as e:
            logger.exception(f"Error clearing collection '{collection_id}': {e}")
            return False

    def get_available_collections(self) -> List[str]:
        if not self._client: return []
        try:
            return [coll.name for coll in self._client.list_collections()]
        except Exception as e:
            logger.error(f"Error listing collections from ChromaDB: {e}", exc_info=True)
            return []

    def delete_collection(self, collection_id: str) -> bool:
        if not self._client: return False
        if collection_id == GLOBAL_COLLECTION_ID:
            logger.warning("Deleting the GLOBAL_COLLECTION is not allowed via this method for safety.")
            return False

        logger.info(f"Deleting collection '{collection_id}' from ChromaDB.")
        try:
            self._client.delete_collection(name=collection_id)
            # AVA_ASSISTANT_MODIFIED: Attempt to remove the directory from disk as well, as ChromaDB might not always clean it up.
            collection_disk_path = Path(self.base_persist_directory) / collection_id
            if collection_disk_path.exists() and collection_disk_path.is_dir():
                try:
                    shutil.rmtree(collection_disk_path)
                    logger.info(f"Successfully removed collection directory from disk: {collection_disk_path}")
                except OSError as e_rm:
                    logger.warning(f"Could not remove collection directory {collection_disk_path} from disk after DB delete: {e_rm}")
            logger.info(f"Collection '{collection_id}' processing for deletion complete.")
            return True
        except Exception as e:
            logger.exception(f"Error deleting collection '{collection_id}': {e}")
            return False