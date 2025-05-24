# services/vector_db_service.py
import logging
import os
import shutil  # For deleting collection directories
import uuid  # For generating unique IDs for ChromaDB documents

# --- ChromaDB Imports ---
try:
    import chromadb

    # Chroma's default embedding function, or use None if we provide embeddings directly.
    # We will provide embeddings directly, so we can use `None` here or pass a specific one.
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore
    CHROMADB_AVAILABLE = False
    logging.critical(
        "VectorDBService: ChromaDB library not found. RAG DB cannot function. Install: pip install chromadb")

from typing import List, Dict, Any, Optional, Tuple

# --- Local Imports ---
from utils import constants

# Define GLOBAL_COLLECTION_ID, typically imported from constants
GLOBAL_COLLECTION_ID = constants.GLOBAL_COLLECTION_ID

logger = logging.getLogger(__name__)


class VectorDBService:
    """
    Manages interactions with ChromaDB, supporting multiple collections (namespaces).
    Each collection represents a distinct logical grouping of documents.
    ChromaDB handles persistence to disk automatically.
    Relies on external embedding generation (embeddings passed to add_embeddings).
    """

    def __init__(self, index_dimension: int, base_persist_directory: Optional[str] = None):
        logger.info("VectorDBService initializing (ChromaDB implementation)...")
        self._client: Optional[chromadb.PersistentClient] = None
        self._index_dim = index_dimension  # Still relevant for consistency check with embeddings

        if not CHROMADB_AVAILABLE:
            logger.critical("ChromaDB library not available. VectorDBService cannot initialize.")
            return

        if not isinstance(index_dimension, int) or index_dimension <= 0:
            logger.critical(f"Invalid index dimension provided: {index_dimension}. Cannot initialize ChromaDB.")
            return

        self.base_persist_directory = base_persist_directory or constants.RAG_COLLECTIONS_PATH
        logger.info(f"Using ChromaDB persist directory: {self.base_persist_directory}")
        try:
            os.makedirs(self.base_persist_directory, exist_ok=True)
            logger.info(f"ChromaDB base directory ensured: {self.base_persist_directory}")
        except OSError as e:
            logger.critical(f"Failed to create ChromaDB base directory '{self.base_persist_directory}': {e}")
            return

        try:
            self._client = chromadb.PersistentClient(path=self.base_persist_directory)
            logger.info("ChromaDB PersistentClient initialized.")

            # Ensure the global collection exists on startup
            if not self.get_or_create_collection(GLOBAL_COLLECTION_ID):
                logger.error(
                    "VectorDBService initialized, but the global collection ('%s') could not be properly created or loaded. "
                    "RAG functionality relying on the global context may be impaired.", GLOBAL_COLLECTION_ID
                )
            else:
                logger.info("VectorDBService initialized successfully and global collection is ready.")

        except Exception as e:
            logger.critical(f"CRITICAL FAILURE: Failed to initialize ChromaDB client: {e}", exc_info=True)
            self._client = None  # Mark as not ready

    def get_or_create_collection(self, collection_id: str) -> Optional[chromadb.Collection]:
        """
        Retrieves an existing ChromaDB collection or creates a new one if it doesn't exist.

        Args:
            collection_id (str): The name of the collection.

        Returns:
            Optional[chromadb.Collection]: The ChromaDB Collection object, or None on error.
        """
        if not CHROMADB_AVAILABLE or self._client is None:
            logger.error(
                f"Cannot get/create collection '{collection_id}': ChromaDB not available or client not initialized.")
            return None
        if not isinstance(collection_id, str) or not collection_id.strip():
            logger.error("Cannot get/create collection: Invalid or empty collection_id provided.")
            return None
        try:
            # When providing embeddings directly via add(), embedding_function can be None
            collection = self._client.get_or_create_collection(
                name=collection_id,
                embedding_function=None  # We will provide embeddings directly when adding
            )
            logger.debug(f"Collection '{collection_id}' accessed/created successfully.")
            return collection
        except Exception as e:
            logger.exception(f"Error getting/creating collection '{collection_id}': {e}")
            return None

    def is_ready(self, collection_id: Optional[str] = None) -> bool:
        """
        Checks if the VectorDBService is ready and, optionally, if a specific collection exists.

        Args:
            collection_id (Optional[str]): If provided, checks for the existence of this collection.
                                           If None, checks if the ChromaDB client is initialized.

        Returns:
            bool: True if ready, False otherwise.
        """
        if not CHROMADB_AVAILABLE or self._client is None:
            return False

        # If no specific collection_id is requested, check if the client itself is initialized.
        # This implicitly means the service is ready to interact with collections.
        if collection_id is None:
            return True

        # If a specific collection_id is requested, check if it exists in the DB.
        try:
            # list_collections() returns a list of Collection objects
            existing_collections = self._client.list_collections()
            return any(coll.name == collection_id for coll in existing_collections)
        except Exception as e:
            logger.error(f"Error checking if collection '{collection_id}' exists: {e}")
            return False

    def add_embeddings(self, collection_id: str, contents: List[str], embeddings: List[List[float]],
                       metadatas: List[Dict[str, Any]]) -> bool:
        """
        Adds content, embeddings, and metadata to a specified ChromaDB collection.

        Args:
            collection_id (str): The ID of the collection to add to.
            contents (List[str]): List of text content for each chunk (Corresponds to Chroma's 'documents').
            embeddings (List[List[float]]): List of embedding vectors (Corresponds to Chroma's 'embeddings').
            metadatas (List[Dict[str, Any]]): List of metadata dictionaries (Corresponds to Chroma's 'metadatas').

        Returns:
            bool: True if successful, False otherwise.
        """
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

        # Validate embedding dimensions
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list) or len(emb) != self._index_dim:
                logger.error(
                    f"Embedding at index {i} has incorrect dimension or type: {len(emb) if isinstance(emb, list) else type(emb)}. Expected {self._index_dim}.")
                return False

        logger.info(f"Adding {len(contents)} documents to collection '{collection_id}'...")

        # Generate unique IDs for each document. ChromaDB can auto-generate, but explicit IDs
        # give us direct control if we ever need to reference them later (e.g., for updates).
        # Using a UUID for uniqueness, combined with a prefix.
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
        """
        Removes documents (chunks) from a collection based on their 'source' metadata field.

        Args:
            collection_id (str): The ID of the collection.
            source_path_to_remove (str): The value of the 'source' metadata field to match.

        Returns:
            bool: True if successful, False otherwise.
        """
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot remove documents: Collection '{collection_id}' not found.")
            return False

        logger.info(f"Removing documents with source '{source_path_to_remove}' from '{collection_id}'...")
        try:
            # ChromaDB's delete method allows filtering by metadata
            collection.delete(where={"source": source_path_to_remove})
            logger.info(f"Successfully deleted documents with source '{source_path_to_remove}' from '{collection_id}'.")
            return True
        except Exception as e:
            logger.exception(f"Error removing documents by source from '{collection_id}': {e}")
            return False

    def search(self, collection_id: str, query_embedding: List[List[float]], k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches a specified ChromaDB collection for similar documents.

        Args:
            collection_id (str): The ID of the collection to search.
            query_embedding (List[List[float]]): A list containing a single query embedding vector.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing 'content', 'metadata', and 'distance'.
        """
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            logger.error(f"Cannot search: Collection '{collection_id}' not found.")
            return []

        if not isinstance(query_embedding, list) or len(query_embedding) != 1 or \
                not isinstance(query_embedding[0], list) or len(query_embedding[0]) != self._index_dim:
            logger.error(
                f"Invalid query embedding format. Expected a list containing one list of {self._index_dim} floats.")
            return []

        if collection.count() == 0:
            logger.debug(f"Collection '{collection_id}' is empty, no results to search.")
            return []

        effective_k = min(k, collection.count())
        if effective_k == 0:
            return []

        results_list = []
        try:
            # ChromaDB's query method returns a dictionary with lists of ids, embeddings, documents, metadatas, and distances.
            # We want documents (text), metadatas, and distances.
            query_results = collection.query(
                query_embeddings=query_embedding,
                n_results=effective_k,
                include=['documents', 'metadatas', 'distances']
            )

            # Reformat the results to match the expected output structure:
            # [{'content': ..., 'metadata': ..., 'distance': ...}]
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
            return results_list
        except Exception as e:
            logger.exception(f"Error searching collection '{collection_id}': {e}")
            return []

    def get_all_metadata(self, collection_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all metadata from a specified ChromaDB collection.
        Note: This can be memory intensive for very large collections.

        Args:
            collection_id (str): The ID of the collection.

        Returns:
            List[Dict[str, Any]]: A list of all metadata dictionaries.
        """
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            return []
        try:
            # Use collection.get to retrieve all documents/metadata.
            # We only need metadatas here.
            all_data = collection.get(ids=None, where=None, include=['metadatas'])
            return all_data.get('metadatas', [])
        except Exception as e:
            logger.exception(f"Error retrieving all metadata from collection '{collection_id}': {e}")
            return []

    def get_collection_size(self, collection_id: str) -> int:
        """
        Returns the number of documents in a specified ChromaDB collection.

        Args:
            collection_id (str): The ID of the collection.

        Returns:
            int: The number of documents, or -1 if the collection is not found.
        """
        collection = self.get_or_create_collection(collection_id)
        if collection is None:
            return -1
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting count for collection '{collection_id}': {e}")
            return -1

    def clear_collection(self, collection_id: str) -> bool:
        """
        Clears all documents from a specified ChromaDB collection.
        This is typically done by deleting and then re-creating the collection.

        Args:
            collection_id (str): The ID of the collection to clear.

        Returns:
            bool: True if successful, False otherwise.
        """
        if collection_id == GLOBAL_COLLECTION_ID:
            logger.warning("Clearing the GLOBAL_COLLECTION is not allowed via this method.")
            return False

        logger.warning(f"Clearing collection '{collection_id}' by deleting and re-creating it.")
        try:
            if self._client:
                self._client.delete_collection(name=collection_id)
                # Re-create to ensure it's ready for new additions
                self.get_or_create_collection(collection_id)  # This call ensures it's re-created
                logger.info(f"Collection '{collection_id}' cleared and re-created successfully.")
                return True
            return False
        except Exception as e:
            logger.exception(f"Error clearing collection '{collection_id}': {e}")
            return False

    def get_available_collections(self) -> List[str]:
        """
        Lists all available collection IDs in the ChromaDB instance.

        Returns:
            List[str]: A list of collection names.
        """
        if not self._client:
            return []
        try:
            return [coll.name for coll in self._client.list_collections()]
        except Exception as e:
            logger.error(f"Error listing collections from ChromaDB: {e}")
            return []

    def delete_collection(self, collection_id: str) -> bool:
        """
        Deletes a specified ChromaDB collection entirely from disk.

        Args:
            collection_id (str): The ID of the collection to delete.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self._client:
            return False
        if collection_id == GLOBAL_COLLECTION_ID:
            logger.warning("Deleting the GLOBAL_COLLECTION is not allowed via this method.")
            return False

        logger.info(f"Deleting collection '{collection_id}' from ChromaDB.")
        try:
            self._client.delete_collection(name=collection_id)
            logger.info(f"Collection '{collection_id}' deleted successfully.")
            return True
        except Exception as e:
            logger.exception(f"Error deleting collection '{collection_id}': {e}")
            return False