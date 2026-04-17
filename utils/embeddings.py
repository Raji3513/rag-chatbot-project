"""
Embeddings Module
Generates vector embeddings using HuggingFace's sentence-transformers.
Uses the 'all-MiniLM-L6-v2' model (free, no API key required).
"""

from langchain_huggingface import HuggingFaceEmbeddings

# Default model — lightweight, fast, and free
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_embeddings(model_name: str = DEFAULT_MODEL):
    """
    Create and return a HuggingFace embeddings instance.
    
    Args:
        model_name (str): The HuggingFace model name to use for embeddings.
                          Default is 'all-MiniLM-L6-v2'.
    
    Returns:
        HuggingFaceEmbeddings: An embeddings model instance.
    """
    print(f"  → Loading embedding model: '{model_name}'")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"  → Embedding model loaded successfully")
    return embeddings