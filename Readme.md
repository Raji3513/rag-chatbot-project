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

## Day 3 – Retrieval & Answer Generation (RAG Pipeline)

### Completed Tasks
- ✅ Retrieval logic (query → embedding → FAISS similarity search)
- ✅ LLM integration using HuggingFace (`google/flan-t5-small`, runs locally)
- ✅ Answer generation using retrieved context (full RAG pipeline)
- ✅ Basic error handling throughout the pipeline
- ✅ Streamlit UI updated with Q&A tab
- ✅ End-to-end RAG: query → retrieve → generate answer

### Features Implemented

#### 5. Document Retrieval (`utils/retriever.py`)
- Searches FAISS vector store using similarity search
- Returns top-k most relevant document chunks
- Input validation and error handling

#### 6. LLM Integration (`utils/llm.py`)
- Uses HuggingFace `google/flan-t5-small` (free, runs locally, no API key)
- Also supports `google/flan-t5-base` for better quality
- Wrapped as LangChain-compatible LLM via `HuggingFacePipeline`

#### 7. RAG Chain (`utils/rag_chain.py`)
- Combines retrieval + LLM into a single `ask_question()` function
- Prompt template instructs LLM to answer based on retrieved context
- Returns answer, sources, and retrieved documents
- Error handling for empty queries, missing index, and LLM failures

#### 8. Updated Streamlit UI (`app/app.py`)
- **Tab 1: Document Ingestion** — Upload and process documents (existing)
- **Tab 2: Ask Questions** — Enter a question, get RAG-generated answers
- Cached LLM and embedding model loading for fast responses
- Configurable LLM model and retrieval count in sidebar

### How to Run

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run the Streamlit App
```bash
streamlit run app/app.py
```
1. Go to **Document Ingestion** tab → upload a document or use sample
2. Go to **Ask Questions** tab → type a question → click "Get Answer"

### Full RAG Pipeline Flow
```
User Question
    │
    ▼
Embedding Model (embeddings.py) ──→ Query Vector
    │
    ▼
FAISS Similarity Search (retriever.py) ──→ Top-K Relevant Chunks
    │
    ▼
Prompt Builder (rag_chain.py) ──→ Context + Question
    │
    ▼
HuggingFace LLM (llm.py) ──→ Generated Answer
    │
    ▼
Display Answer + Sources (app.py)
```

---