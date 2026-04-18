"""
Retriever Module
Handles query → vector DB search using FAISS.
Returns relevant document chunks for a given query.
"""


def retrieve_documents(vector_store, query, k=3):
    """
    Search the FAISS vector store and return relevant documents.

    Args:
        vector_store: A loaded FAISS vector store instance.
        query (str): The user's question or search query.
        k (int): Number of top results to return.

    Returns:
        list: A list of relevant Document objects.

    Raises:
        ValueError: If the query is empty.
        ValueError: If vector_store is None.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if vector_store is None:
        raise ValueError("Vector store is not loaded. Please ingest documents first.")

    results = vector_store.similarity_search(query, k=k)
    print("  -> Retrieved {} document(s) for query: '{}'".format(len(results), query[:50]))
    return results
