"""
RAG Chatbot – Day 2: Document Upload & Preprocessing
Streamlit web application for uploading documents (PDF/TXT),
splitting them into chunks, generating embeddings, and storing in FAISS.
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

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot – Document Ingestion",
    page_icon="📄",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="main-header">📄 RAG Chatbot – Document Ingestion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload PDF or TXT documents → Split into chunks → Generate embeddings → Store in FAISS vector database</div>', unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────
# Sidebar — Configuration
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    chunk_size = st.slider("Chunk Size (characters)", 100, 2000, 500, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 200, 50, step=10)
    embedding_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
        index=0
    )

    st.divider()
    st.header("📊 Pipeline Info")
    st.markdown("""
    **Steps:**
    1. 📤 Upload document
    2. ✂️ Split into chunks
    3. 🧠 Generate embeddings
    4. 💾 Store in FAISS
    """)

    st.divider()
    st.caption("RAG Chatbot Project – Day 2")

# ──────────────────────────────────────────────
# Main Content — File Upload
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# Processing Pipeline
# ──────────────────────────────────────────────
if uploaded_file is not None or use_sample:
    st.divider()
    st.subheader("🔄 Processing Pipeline")

    progress_bar = st.progress(0, text="Starting pipeline...")
    status_container = st.container()

    try:
        # ── Step 1: Load Document ──
        with status_container:
            with st.spinner("📥 Loading document..."):
                progress_bar.progress(10, text="Step 1/4: Loading document...")

                if uploaded_file is not None:
                    documents = load_uploaded_file(uploaded_file)
                    file_name = uploaded_file.name
                    file_size = uploaded_file.size
                else:
                    # Use the sample file
                    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    sample_path = os.path.join(BASE_DIR, "data", "sample.txt")
                    documents = load_document(sample_path)
                    file_name = "sample.txt"
                    file_size = os.path.getsize(sample_path)

                st.success(f"✅ **Step 1 Complete**: Loaded `{file_name}` — {len(documents)} page(s)/section(s)")
                progress_bar.progress(25, text="Step 1/4: Document loaded ✓")

        # ── Step 2: Split into Chunks ──
        with status_container:
            with st.spinner("✂️ Splitting document into chunks..."):
                progress_bar.progress(30, text="Step 2/4: Splitting document...")
                chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.success(f"✅ **Step 2 Complete**: Created `{len(chunks)}` chunks (size={chunk_size}, overlap={chunk_overlap})")
                progress_bar.progress(50, text="Step 2/4: Document split ✓")

        # ── Step 3: Generate Embeddings ──
        with status_container:
            with st.spinner(f"🧠 Loading embedding model: `{embedding_model}`..."):
                progress_bar.progress(55, text="Step 3/4: Generating embeddings...")
                embeddings = get_embeddings(model_name=embedding_model)
                st.success(f"✅ **Step 3 Complete**: Embedding model `{embedding_model}` loaded")
                progress_bar.progress(75, text="Step 3/4: Embeddings ready ✓")

        # ── Step 4: Store in FAISS ──
        with status_container:
            with st.spinner("💾 Creating and saving FAISS vector store..."):
                progress_bar.progress(80, text="Step 4/4: Storing in FAISS...")
                vector_store = create_vector_store(chunks, embeddings)
                st.success(f"✅ **Step 4 Complete**: FAISS vector store created and saved to `embeddings/`")
                progress_bar.progress(100, text="Pipeline complete! ✓")

        # ── Success Summary ──
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

        # ── Show Chunk Preview ──
        st.divider()
        st.subheader("🔍 Chunk Preview")

        preview_count = min(5, len(chunks))
        tabs = st.tabs([f"Chunk {i+1}" for i in range(preview_count)])
        for i, tab in enumerate(tabs):
            with tab:
                st.markdown(f"**Chunk {i+1}** — {len(chunks[i].page_content)} characters")
                st.code(chunks[i].page_content, language=None)
                if chunks[i].metadata:
                    st.caption(f"Metadata: {chunks[i].metadata}")

        # ── Test Similarity Search ──
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

# ──────────────────────────────────────────────
# Footer — FAISS Index Status
# ──────────────────────────────────────────────
st.divider()
st.subheader("💾 FAISS Index Status")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
embeddings_dir = os.path.join(BASE_DIR, "embeddings")
index_file = os.path.join(embeddings_dir, "index.faiss")
pkl_file = os.path.join(embeddings_dir, "index.pkl")

if os.path.exists(index_file) and os.path.exists(pkl_file):
    faiss_size = os.path.getsize(index_file)
    pkl_size = os.path.getsize(pkl_file)
    st.success(f"✅ FAISS index exists at `embeddings/`")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("index.faiss", f"{faiss_size / 1024:.1f} KB")
    with col_b:
        st.metric("index.pkl", f"{pkl_size / 1024:.1f} KB")
else:
    st.info("ℹ️ No FAISS index found yet. Upload a document to create one.")
