"""
Vector Store Module
Creates and persists a FAISS vector store from document chunks and embeddings.
Supports saving to and loading from disk for persistence.
"""

import os
from langchain_community.vectorstores import FAISS
# Default directory to persist the FAISS index
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings")


def create_vector_store(chunks, embeddings, persist_directory=None):
    """
    Create a FAISS vector store from document chunks and save it to disk.

    Args:
        chunks (list): List of chunked Document objects.
        embeddings: An embeddings model instance.
        persist_directory (str): Directory path to save the FAISS index.

    Returns:
        FAISS: The created FAISS vector store.
    """
    if persist_directory is None:
        persist_directory = DEFAULT_PERSIST_DIR

    print("  -> Creating FAISS vector store from {} chunks...".format(len(chunks)))
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    # Save the FAISS index to disk
    vector_store.save_local(persist_directory)
    print("  -> FAISS index saved to: {}".format(persist_directory))

    return vector_store


def load_vector_store(embeddings, persist_directory=None):
    """
    Load a previously saved FAISS vector store from disk.

    Args:
        embeddings: An embeddings model instance (must match the one used to create the store).
        persist_directory (str): Directory path where the FAISS index is saved.

    Returns:
        FAISS: The loaded FAISS vector store, or None if not found.
    """
    if persist_directory is None:
        persist_directory = DEFAULT_PERSIST_DIR

    index_path = os.path.join(persist_directory, "index.faiss")
    if not os.path.exists(index_path):
        print("  -> No existing FAISS index found at: {}".format(persist_directory))
        return None

    print("  -> Loading FAISS index from: {}".format(persist_directory))
    vector_store = FAISS.load_local(
        persist_directory, embeddings, allow_dangerous_deserialization=True
    )
    print("  -> FAISS index loaded successfully")
    return vector_store