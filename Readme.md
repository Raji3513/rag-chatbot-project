# RAG Chatbot Project

A Retrieval-Augmented Generation (RAG) chatbot that processes documents, generates embeddings, and enables intelligent question-answering.

---

## Day 1 – Setup & Planning

### Completed Tasks
- Set up project repository
- Created project structure
- Initialized Git
- Planned architecture

### Project Structure
```
rag-chatbot-project/
├── app/
│   └── app.py              # Streamlit web UI for document upload
├── data/
│   └── sample.txt           # Sample document for testing
├── embeddings/              # FAISS vector store (auto-generated)
│   ├── index.faiss
│   └── index.pkl
├── utils/
│   ├── __init__.py
│   ├── loader.py            # Document loader (PDF/TXT)
│   ├── splitter.py          # Text chunking
│   ├── embeddings.py        # HuggingFace embeddings
│   └── vector_store.py      # FAISS create/save/load
├── Screenshots/             # Proof screenshots
├── main.py                  # CLI pipeline script
├── requirements.txt         # Python dependencies
└── README.md
```

### Tech Stack
- Python 3.10+
- LangChain
- FAISS (Vector Store)
- Streamlit (Web UI)
- HuggingFace Sentence Transformers (Embeddings)

---

## Day 2 – Document Upload & Preprocessing

### Completed Tasks
- ✅ Document ingestion (PDF/TXT upload via Streamlit web UI)
- ✅ Document splitting into chunks (configurable size & overlap)
- ✅ Embedding generation using HuggingFace `all-MiniLM-L6-v2`
- ✅ Vector storage in FAISS with persistence (save/load to disk)
- ✅ Similarity search verification
- ✅ CLI pipeline script for batch processing

### Features Implemented

#### 1. Document Loading (`utils/loader.py`)
- Supports **PDF** (via PyPDFLoader) and **TXT** (via TextLoader) formats
- Handles both file-path and Streamlit file-upload objects
- Automatic temp file management for uploads
- Source metadata tagging

#### 2. Text Chunking (`utils/splitter.py`)
- Uses `RecursiveCharacterTextSplitter` for intelligent splitting
- Configurable `chunk_size` (default: 500) and `chunk_overlap` (default: 50)
- Smart separators: `\n\n`, `\n`, `. `, ` `, ``

#### 3. Embedding Generation (`utils/embeddings.py`)
- Uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Free, no API key required
- Supports model selection (MiniLM, mpnet, paraphrase)

#### 4. FAISS Vector Store (`utils/vector_store.py`)
- Creates FAISS index from document chunks
- **Persists to disk** (`embeddings/` directory)
- Load existing index for reuse
- Supports similarity search queries

#### 5. Streamlit Web UI (`app/app.py`)
- Upload PDF/TXT files via drag-and-drop
- Configurable chunking parameters (sidebar)
- Real-time progress tracking
- Chunk preview with metadata
- Built-in similarity search testing
- FAISS index status display

### How to Run

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Option 1: CLI Pipeline
```bash
python main.py                     # Process sample.txt
python main.py path/to/file.pdf    # Process a specific file
```

#### Option 2: Streamlit Web UI
```bash
streamlit run app/app.py
```

### Pipeline Flow
```
Document (PDF/TXT)
    │
    ▼
Document Loader (loader.py)
    │
    ▼
Text Splitter (splitter.py) ──→ Chunks
    │
    ▼
Embedding Model (embeddings.py) ──→ Vectors
    │
    ▼
FAISS Vector Store (vector_store.py) ──→ Persisted to disk
    │
    ▼
Similarity Search ──→ Relevant Results
```

---