"""
Document Splitter Module
Splits loaded documents into smaller chunks for embedding generation.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into smaller chunks.
    
    Args:
        documents (list): List of LangChain Document objects.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    
    Returns:
        list: A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  → Split into {len(chunks)} chunk(s) (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks