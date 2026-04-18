"""
RAG Chatbot — Full Pipeline (Day 3)
Streamlit UI: Tab 1 = Document Ingestion | Tab 2 = Ask Questions
"""
import os, sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loader      import load_uploaded_file, load_document
from utils.splitter    import split_documents
from utils.embeddings  import get_embeddings
from utils.vector_store import create_vector_store, load_vector_store
from utils.llm         import get_llm
from utils.rag_chain   import ask_question

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="MindX RAG — Intelligent Document AI",
                   page_icon="🧠", layout="wide")

# ── Load external CSS ────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_llm_cached(m):       return get_llm(model_name=m)
@st.cache_resource
def load_embed_cached(m):     return get_embeddings(model_name=m)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR       = os.path.join(BASE_DIR, "embeddings")
INDEX_FILE    = os.path.join(EMB_DIR, "index.faiss")
PKL_FILE      = os.path.join(EMB_DIR, "index.pkl")
faiss_ready   = os.path.exists(INDEX_FILE) and os.path.exists(PKL_FILE)

# ── Helper: render metric pills ─────────────────────────────────────────────
def metric_pills(items):
    cols = st.columns(len(items))
    for col, (key, val) in zip(cols, items):
        with col:
            st.markdown(f'<div class="metric-pill"><span class="val">{val}</span>'
                        f'<span class="key">{key}</span></div>', unsafe_allow_html=True)

# ── Helper: render context chunks ───────────────────────────────────────────
def render_context(docs):
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "N/A")
        st.markdown(f'<div class="context-pill"><div class="context-badge">Chunk {i+1} · {src}</div>'
                    f'<br>{doc.page_content[:400]}{"…" if len(doc.page_content)>400 else ""}'
                    f'</div>', unsafe_allow_html=True)

# ── Banner ───────────────────────────────────────────────────────────────────
st.markdown('<div class="banner">🚀 MindX RAG — HuggingFace + FAISS, 100% local, no API keys needed'
            ' <em>Try It Now ➔</em></div>', unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Retrieval-Augmented Generation</div>
    <h1>Turn Every Document Into<br>an <span class="green">Intelligent Answer</span></h1>
    <p>Upload any PDF or TXT. We split, embed &amp; store it — then answer your questions
       with context-grounded responses. No hallucinations. Just facts from your data.</p>
    <div class="hero-stats">
        <div class="hero-stat"><span class="num">4-Step</span><span class="lbl">Ingestion Pipeline</span></div>
        <div class="hero-stat"><span class="num">FAISS</span><span class="lbl">Vector Search</span></div>
        <div class="hero-stat"><span class="num">Flan-T5</span><span class="lbl">Local LLM</span></div>
        <div class="hero-stat"><span class="num">0 API</span><span class="lbl">Keys Required</span></div>
    </div>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 MindX RAG")
    st.markdown("<small>Intelligent Document Q&A</small>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**⚙️ Ingestion Settings**")
    chunk_size      = st.slider("Chunk Size (chars)", 100, 2000, 500, 50)
    chunk_overlap   = st.slider("Chunk Overlap (chars)", 0, 200, 50, 10)
    embedding_model = st.selectbox("Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"])
    st.divider()
    st.markdown("**🤖 LLM Settings**")
    llm_model   = st.selectbox("LLM Model", ["google/flan-t5-small", "google/flan-t5-base"])
    num_results = st.slider("Results to retrieve (k)", 1, 10, 3)
    st.divider()
    st.markdown("**📊 RAG Pipeline**")
    for icon, title, desc in [("📤","Upload","PDF or TXT"), ("✂️","Split","Into chunks"),
                                ("🧠","Embed","HuggingFace"), ("💾","Store","FAISS DB"),
                                ("🔍","Retrieve","Similarity search"), ("💬","Answer","Flan-T5 LLM")]:
        st.markdown(f'<div class="step-row"><div class="step-num">{icon}</div>'
                    f'<div class="step-content"><h4>{title}</h4><p>{desc}</p></div></div>',
                    unsafe_allow_html=True)
    st.divider()
    st.caption("MindX RAG Chatbot — Day 3 Build")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄  Document Ingestion", "💬  Ask Questions"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ingestion
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([3, 1], gap="large")
    with c1:
        st.markdown('<div class="upload-zone"><div class="icon">📂</div>'
                    '<h3>Drop your document here</h3>'
                    '<p>PDF or TXT — processed entirely on your machine</p></div>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader("File", type=["pdf","txt"],
                                         label_visibility="collapsed")
    with c2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        use_sample = st.button("🗂️ Use Sample Document", use_container_width=True)

    if uploaded_file or use_sample:
        st.markdown("---")
        st.markdown("### 🔄 Running Pipeline")
        bar = st.progress(0, text="Initialising…")
        try:
            # Step 1 — Load
            with st.spinner("📥 Loading…"):
                bar.progress(10, "Step 1/4 — Loading…")
                if uploaded_file:
                    docs = load_uploaded_file(uploaded_file)
                    fname, fsize = uploaded_file.name, uploaded_file.size
                else:
                    sp = os.path.join(BASE_DIR, "data", "sample.txt")
                    docs = load_document(sp)
                    fname, fsize = "sample.txt", os.path.getsize(sp)
                st.success(f"✅ Loaded `{fname}` — {len(docs)} section(s)")
                bar.progress(25, "Step 1/4 — Done ✓")

            # Step 2 — Split
            with st.spinner("✂️ Splitting…"):
                bar.progress(30, "Step 2/4 — Splitting…")
                chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.success(f"✅ Created `{len(chunks)}` chunks")
                bar.progress(50, "Step 2/4 — Done ✓")

            # Step 3 — Embed
            with st.spinner(f"🧠 Loading `{embedding_model}`…"):
                bar.progress(55, "Step 3/4 — Embedding…")
                embeddings = get_embeddings(model_name=embedding_model)
                st.success(f"✅ Embedding model ready")
                bar.progress(75, "Step 3/4 — Done ✓")

            # Step 4 — FAISS
            with st.spinner("💾 Building FAISS index…"):
                bar.progress(80, "Step 4/4 — Storing…")
                vector_store = create_vector_store(chunks, embeddings)
                st.success("✅ FAISS index saved to `embeddings/`")
                bar.progress(100, "Pipeline complete ✓")

            # Summary
            st.markdown("---")
            st.markdown("### 🎉 Ingestion Complete")
            metric_pills([("📄 File", fname), ("📦 Sections", len(docs)),
                          ("✂️ Chunks", len(chunks)), ("🧠 Model", embedding_model.split("-")[0])])

            # Chunk preview
            st.markdown("---")
            st.markdown("### 🔍 Chunk Preview")
            for i, ctab in enumerate(st.tabs([f"Chunk {i+1}" for i in range(min(5,len(chunks)))])):
                with ctab:
                    st.caption(f"{len(chunks[i].page_content)} characters · {chunks[i].metadata}")
                    st.code(chunks[i].page_content, language=None)

            # Test search
            st.markdown("---")
            st.markdown("### 🔎 Test Similarity Search")
            q = st.text_input("Search query:", placeholder="What is this document about?", key="test_q")
            if q:
                results = vector_store.similarity_search(q, k=3)
                render_context(results)

        except Exception as e:
            st.error(f"❌ {e}"); st.exception(e)

    # FAISS status
    st.markdown("---")
    if faiss_ready:
        fs, ps = os.path.getsize(INDEX_FILE), os.path.getsize(PKL_FILE)
        st.markdown(f'<div class="faiss-ok"><div style="font-size:2rem">✅</div>'
                    f'<div><h4>Vector index is ready</h4>'
                    f'<p>index.faiss {fs/1024:.1f} KB &nbsp;|&nbsp; index.pkl {ps/1024:.1f} KB</p>'
                    f'</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="faiss-empty">⚠️ No FAISS index yet — upload a document above.</div>',
                    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Q&A
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="glass-card"><div style="display:flex;align-items:center;gap:12px">'
                '<div style="font-size:2rem">💬</div>'
                '<div><h3 style="margin:0;color:#0f172a">Ask Anything About Your Documents</h3>'
                '<p style="margin:0;color:#64748b;font-size:.9rem">FAISS retrieves context → Flan-T5 generates a grounded answer.</p>'
                '</div></div></div>', unsafe_allow_html=True)

    if not faiss_ready:
        st.markdown('<div class="faiss-empty">⚠️ No documents ingested yet — go to Document Ingestion first.</div>',
                    unsafe_allow_html=True)
    else:
        if "history" not in st.session_state:
            st.session_state.history = []

        ic, bc = st.columns([5, 1], gap="small")
        with ic:
            question = st.text_input("Question:", placeholder="e.g. What is the main topic?",
                                     label_visibility="collapsed", key="rag_q")
        with bc:
            ask_btn = st.button("🔍 Ask", use_container_width=True)

        if ask_btn and question.strip():
            try:
                with st.spinner("Loading models & searching…"):
                    emb  = load_embed_cached(embedding_model)
                    vs   = load_vector_store(emb)
                    llm  = load_llm_cached(llm_model)
                    res  = ask_question(vs, llm, question, k=num_results)
                st.session_state.history.append(
                    {"q": question, "a": res["answer"], "docs": res["documents"]})
            except Exception as e:
                st.error(f"❌ {e}"); st.exception(e)
        elif ask_btn:
            st.warning("Please type a question first.")

        # Chat history
        if st.session_state.history:
            st.markdown("---")
            for entry in reversed(st.session_state.history):
                st.markdown(f'<div class="chat-question"><div class="bubble">🧑 &nbsp;{entry["q"]}</div></div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="chat-answer"><div class="chat-avatar">🧠</div>'
                            f'<div class="bubble">{entry["a"]}</div></div>', unsafe_allow_html=True)
                if entry["docs"]:
                    with st.expander(f"📚 {len(entry['docs'])} source chunk(s)"):
                        render_context(entry["docs"])
                st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:.6rem 0'>",
                            unsafe_allow_html=True)

            if st.button("🗑️ Clear Conversation"):
                st.session_state.history = []; st.rerun()

        st.markdown(f"<small style='color:#94a3b8'>LLM: <code>{llm_model}</code> &nbsp;|&nbsp; "
                    f"Embedding: <code>{embedding_model}</code> &nbsp;|&nbsp; Top-<code>{num_results}</code></small>",
                    unsafe_allow_html=True)
