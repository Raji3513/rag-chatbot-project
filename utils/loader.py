"""
Document Loader Module
Handles loading of PDF and TXT files using LangChain document loaders.
Supports both file-path-based loading and in-memory (uploaded) file loading.
"""

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(file_path: str):
    """
    Load a document from a file path.
    Supports .pdf and .txt formats.
    
    Args:
        file_path (str): Absolute path to the document file.
    
    Returns:
        list: A list of LangChain Document objects.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")

    documents = loader.load()
    print(f"  → Loaded {len(documents)} page(s)/section(s) from '{os.path.basename(file_path)}'")
    return documents


def load_uploaded_file(uploaded_file):
    """
    Load a document from a Streamlit UploadedFile object.
    Saves to a temp file, then uses the standard loader.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        list: A list of LangChain Document objects.
    """
    # Determine the file extension
    file_name = uploaded_file.name
    suffix = os.path.splitext(file_name)[1].lower()

    if suffix not in [".pdf", ".txt"]:
        raise ValueError(f"Unsupported file format: {suffix}. Please upload a .pdf or .txt file.")

    # Write uploaded bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        documents = load_document(tmp_path)
        # Attach original filename as metadata
        for doc in documents:
            doc.metadata["source"] = file_name
        return documents
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)