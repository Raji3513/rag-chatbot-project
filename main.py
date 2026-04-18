"""
RAG Chatbot - Day 2: Main Pipeline Script
Processes documents from the data/ folder through the full RAG ingestion pipeline:
  1. Load document (PDF/TXT)
  2. Split into chunks
  3. Generate embeddings (HuggingFace)
  4. Store in FAISS vector store

Usage:
    python main.py                      # Process default sample.txt
    python main.py path/to/file.pdf     # Process a specific file
"""

import os
import sys

from utils.loader import load_document
from utils.splitter import split_documents
from utils.embeddings import get_embeddings
from utils.vector_store import create_vector_store


def main(file_path=None):
    """Run the full document ingestion pipeline."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Determine which file to process
    if file_path is None:
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            file_path = os.path.join(BASE_DIR, "data", "sample.txt")

    print("=" * 60)
    print("  RAG Chatbot - Document Ingestion Pipeline")
    print("=" * 60)
    print("\n  File: {}\n".format(file_path))

    # Step 1: Load
    print("-" * 40)
    print("Step 1: Loading document...")
    documents = load_document(file_path)
    print("  [OK] Loaded {} page(s)/section(s)\n".format(len(documents)))

    # Step 2: Split
    print("-" * 40)
    print("Step 2: Splitting document into chunks...")
    chunks = split_documents(documents)
    print("  [OK] Created {} chunk(s)\n".format(len(chunks)))

    # Preview first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.page_content[:100].replace("\n", " ")
        print("  [Chunk {}] ({} chars): {}...".format(i + 1, len(chunk.page_content), preview))

    # Step 3: Embeddings
    print("\n" + "-" * 40)
    print("Step 3: Generating embeddings...")
    embeddings = get_embeddings()
    print("  [OK] Embedding model loaded\n")

    # Step 4: Vector Store
    print("-" * 40)
    print("Step 4: Creating and saving FAISS vector store...")
    vector_store = create_vector_store(chunks, embeddings)
    print("  [OK] FAISS index created and persisted\n")

    # Verification - test similarity search
    print("-" * 40)
    print("Step 5: Verification - Testing similarity search...")
    test_query = "What is this document about?"
    results = vector_store.similarity_search(test_query, k=2)
    print('  Query: "{}"'.format(test_query))
    for i, res in enumerate(results):
        preview = res.page_content[:120].replace("\n", " ")
        print("  [Result {}]: {}...".format(i + 1, preview))

    # Final summary
    embeddings_dir = os.path.join(BASE_DIR, "embeddings")
    index_file = os.path.join(embeddings_dir, "index.faiss")
    pkl_file = os.path.join(embeddings_dir, "index.pkl")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE - Summary")
    print("=" * 60)
    print("  Document:        {}".format(os.path.basename(file_path)))
    print("  Pages/Sections:  {}".format(len(documents)))
    print("  Chunks:          {}".format(len(chunks)))
    print("  Embedding Model: all-MiniLM-L6-v2")
    print("  FAISS Index:     {}".format(embeddings_dir))
    if os.path.exists(index_file):
        print("     - index.faiss:  {:.1f} KB".format(os.path.getsize(index_file) / 1024))
    if os.path.exists(pkl_file):
        print("     - index.pkl:    {:.1f} KB".format(os.path.getsize(pkl_file) / 1024))
    print("=" * 60)
    print("\n  SUCCESS! Document processed and stored in FAISS.")
    print("  You can now query this data using similarity search.\n")


if __name__ == "__main__":
    main()