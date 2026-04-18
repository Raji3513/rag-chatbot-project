"""
RAG Chatbot – Full Pipeline (Day 3)
Streamlit web application with two tabs:
  Tab 1: Document ingestion (upload, chunk, embed, store in FAISS)
  Tab 2: Question answering (retrieve context, generate answer with LLM)
"""

import os
import sys
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loader import load_uploaded_file, load_document
from utils.splitter import split_documents
from utils.embeddings import get_embeddings
from utils.vector_store import create_vector_store, load_vector_store
from utils.retriever import retrieve_documents
from utils.llm import get_llm
from utils.rag_chain import ask_question

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ──────────────────────────────────────────────
# Cached Resource Loaders
# ──────────────────────────────────────────────
@st.cache_resource
def load_llm_cached(model_name):
    """Cache the LLM so it only loads once."""
    return get_llm(model_name=model_name)

@st.cache_resource
def load_embeddings_cached(model_name):
    """Cache the embeddings model so it only loads once."""
    return get_embeddings(model_name=model_name)

# ──────────────────────────────────────────────
# Custom Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="main-header">🤖 RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload documents and ask questions using Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar — Configuration
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📄 Ingestion Settings")
    chunk_size = st.slider("Chunk Size (characters)", 100, 2000, 500, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 200, 50, step=10)
    embedding_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
        index=0
    )

    st.divider()
    st.subheader("🤖 LLM Settings")
    llm_model = st.selectbox(
        "LLM Model",
        ["google/flan-t5-small", "google/flan-t5-base"],
        index=0
    )
    num_results = st.slider("Results to retrieve", 1, 10, 3)

    st.divider()
    st.header("📊 Pipeline Info")
    st.markdown("""
    **RAG Pipeline:**
    1. 📤 Upload document
    2. ✂️ Split into chunks
    3. 🧠 Generate embeddings
    4. 💾 Store in FAISS
    5. 🔍 Retrieve context
    6. 💬 Generate answer
    """)

    st.divider()
    st.caption("RAG Chatbot Project – Day 3")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
embeddings_dir = os.path.join(BASE_DIR, "embeddings")
index_file = os.path.join(embeddings_dir, "index.faiss")
pkl_file = os.path.join(embeddings_dir, "index.pkl")

# ──────────────────────────────────────────────
# Main Content — Tabs
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄 Document Ingestion", "💬 Ask Questions"])

# ══════════════════════════════════════════════
# Tab 1: Document Ingestion
# ══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
            help="Supported formats: PDF (.pdf) and Text (.txt)"
        )

    with col2:
        st.subheader("📁 Or Use Sample")
        use_sample = st.button("🗂️ Use Sample Document", use_container_width=True)

    # ── Processing Pipeline ──
    if uploaded_file is not None or use_sample:
        st.divider()
        st.subheader("🔄 Processing Pipeline")

        progress_bar = st.progress(0, text="Starting pipeline...")
        status_container = st.container()

        try:
            # Step 1: Load Document
            with status_container:
                with st.spinner("📥 Loading document..."):
                    progress_bar.progress(10, text="Step 1/4: Loading document...")

                    if uploaded_file is not None:
                        documents = load_uploaded_file(uploaded_file)
                        file_name = uploaded_file.name
                        file_size = uploaded_file.size
                    else:
                        sample_path = os.path.join(BASE_DIR, "data", "sample.txt")
                        documents = load_document(sample_path)
                        file_name = "sample.txt"
                        file_size = os.path.getsize(sample_path)

                    st.success(f"✅ **Step 1 Complete**: Loaded `{file_name}` — {len(documents)} page(s)/section(s)")
                    progress_bar.progress(25, text="Step 1/4: Document loaded ✓")

            # Step 2: Split into Chunks
            with status_container:
                with st.spinner("✂️ Splitting document into chunks..."):
                    progress_bar.progress(30, text="Step 2/4: Splitting document...")
                    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.success(f"✅ **Step 2 Complete**: Created `{len(chunks)}` chunks (size={chunk_size}, overlap={chunk_overlap})")
                    progress_bar.progress(50, text="Step 2/4: Document split ✓")

            # Step 3: Generate Embeddings
            with status_container:
                with st.spinner(f"🧠 Loading embedding model: `{embedding_model}`..."):
                    progress_bar.progress(55, text="Step 3/4: Generating embeddings...")
                    embeddings = get_embeddings(model_name=embedding_model)
                    st.success(f"✅ **Step 3 Complete**: Embedding model `{embedding_model}` loaded")
                    progress_bar.progress(75, text="Step 3/4: Embeddings ready ✓")

            # Step 4: Store in FAISS
            with status_container:
                with st.spinner("💾 Creating and saving FAISS vector store..."):
                    progress_bar.progress(80, text="Step 4/4: Storing in FAISS...")
                    vector_store = create_vector_store(chunks, embeddings)
                    st.success(f"✅ **Step 4 Complete**: FAISS vector store created and saved to `embeddings/`")
                    progress_bar.progress(100, text="Pipeline complete! ✓")

            # Success Summary
            st.divider()
            st.subheader("🎉 Ingestion Complete!")

            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("📄 File", file_name)
            with metric_cols[1]:
                st.metric("📦 Pages/Sections", len(documents))
            with metric_cols[2]:
                st.metric("✂️ Chunks", len(chunks))
            with metric_cols[3]:
                st.metric("🧠 Model", embedding_model)

            # Show Chunk Preview
            st.divider()
            st.subheader("🔍 Chunk Preview")

            preview_count = min(5, len(chunks))
            chunk_tabs = st.tabs([f"Chunk {i+1}" for i in range(preview_count)])
            for i, ctab in enumerate(chunk_tabs):
                with ctab:
                    st.markdown(f"**Chunk {i+1}** — {len(chunks[i].page_content)} characters")
                    st.code(chunks[i].page_content, language=None)
                    if chunks[i].metadata:
                        st.caption(f"Metadata: {chunks[i].metadata}")

            # Test Similarity Search
            st.divider()
            st.subheader("🔎 Test Similarity Search")

            query = st.text_input("Enter a test query to search the vector store:", placeholder="e.g., What is this document about?")

            if query:
                with st.spinner("Searching..."):
                    results = vector_store.similarity_search(query, k=3)
                    st.write(f"**Top {len(results)} results:**")
                    for idx, result in enumerate(results):
                        with st.expander(f"Result {idx + 1} (Source: {result.metadata.get('source', 'N/A')})"):
                            st.write(result.page_content)

        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)

    # Footer — FAISS Index Status
    st.divider()
    st.subheader("💾 FAISS Index Status")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        faiss_size = os.path.getsize(index_file)
        pkl_size = os.path.getsize(pkl_file)
        st.success("✅ FAISS index exists at `embeddings/`")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("index.faiss", f"{faiss_size / 1024:.1f} KB")
        with col_b:
            st.metric("index.pkl", f"{pkl_size / 1024:.1f} KB")
    else:
        st.info("ℹ️ No FAISS index found yet. Upload a document to create one.")

# ══════════════════════════════════════════════
# Tab 2: Ask Questions (RAG Q&A)
# ══════════════════════════════════════════════
with tab2:
    st.subheader("💬 Ask Questions About Your Documents")
    st.markdown("Ask any question and the RAG pipeline will retrieve relevant context and generate an answer using a HuggingFace LLM.")

    # Check if FAISS index exists
    if not os.path.exists(index_file) or not os.path.exists(pkl_file):
        st.warning("⚠️ No documents ingested yet. Please go to the **Document Ingestion** tab and upload a document first.")
    else:
        # Question input
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What is reinforcement learning?",
            key="rag_question"
        )

        if st.button("🔍 Get Answer", use_container_width=True, type="primary"):
            if not question or not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    # Step 1: Load embeddings and vector store
                    with st.spinner("🔄 Loading embedding model..."):
                        embeddings = load_embeddings_cached(embedding_model)
                        vector_store = load_vector_store(embeddings)

                    if vector_store is None:
                        st.error("❌ Failed to load FAISS index. Please re-ingest your documents.")
                    else:
                        # Step 2: Load LLM
                        with st.spinner(f"🤖 Loading LLM: `{llm_model}` (first time may take a moment)..."):
                            llm = load_llm_cached(llm_model)

                        # Step 3: Run RAG pipeline
                        with st.spinner("🔍 Retrieving context and generating answer..."):
                            result = ask_question(vector_store, llm, question, k=num_results)

                        # Display answer
                        st.divider()
                        st.subheader("📝 Answer")
                        st.markdown(
                            '<div class="answer-box">{}</div>'.format(result["answer"]),
                            unsafe_allow_html=True
                        )

                        # Display retrieved context
                        st.divider()
                        st.subheader("📚 Retrieved Context ({} chunks)".format(len(result["documents"])))
                        for i, doc in enumerate(result["documents"]):
                            with st.expander("Chunk {} (Source: {})".format(i + 1, doc.metadata.get("source", "N/A"))):
                                st.write(doc.page_content)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

        # Show current config
        st.divider()
        st.caption(f"LLM: `{llm_model}` | Embedding: `{embedding_model}` | Top-{num_results} retrieval")
