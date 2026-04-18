"""
Embeddings Module
Generates vector embeddings using HuggingFace's sentence-transformers.
Uses the 'all-MiniLM-L6-v2' model (free, no API key required).
"""

# Try the modern package first, fall back to community package
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Default model - lightweight, fast, and free
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_embeddings(model_name=DEFAULT_MODEL):
    """
    Create and return a HuggingFace embeddings instance.

    Args:
        model_name (str): The HuggingFace model name to use for embeddings.
                          Default is 'all-MiniLM-L6-v2'.

    Returns:
        HuggingFaceEmbeddings: An embeddings model instance.
    """
    print("  -> Loading embedding model: '{}'".format(model_name))
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("  -> Embedding model loaded successfully")
    return embeddings