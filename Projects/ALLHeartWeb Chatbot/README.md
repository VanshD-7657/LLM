# WHOIS Data Center RAG Customer Support AI Assistant

A production-grade, highly modular Retrieval-Augmented Generation (RAG) Customer Support system built to answer both general company questions and technical API documentation questions for the WHOIS Data Center platform.

---

## Architecture Overview

```text
User Query
    ↓
Query Contextualization (Conversational Memory)
    ↓
Intent Classification (Query Router)
    ↓
Retriever Selection
    ↓
Hybrid Retrieval (Semantic + BM25) in ChromaDB
    ↓
Reciprocal Rank Fusion (RRF Re-ranking)
    ↓
Context & Prompt Construction (with Source References)
    ↓
LLM Execution (Gemini / Swappable Providers)
    ↓
Answer & Interactive Citations
```

### Component Details

1. **Ingestion Pipeline (`app/ingestion.py`)**: 
   - Loads Markdown documents (`docs/`) and plain text files (`knowledge/`), excluding `combined_reference.md` to prevent redundancy.
   - Extracts structured YAML frontmatter metadata for API documents and URL/metadata headers for company documents.
   - **Chunking Strategy**: Implements semantic block-based chunking. It parses text line-by-line into structural blocks (headers, paragraphs, tables, and code fences). Chunks are assembled within **800–1200 characters** with a **150–200 characters sliding window overlap**, strictly ensuring code blocks, endpoint panels, and parameter tables are **never split**.

2. **Embedding & Vector DB Layer (`app/embeddings.py`, `app/db.py`)**:
   - Connects to **ChromaDB** as the local vector database.
   - Implements swappable embedding providers. It defaults to **Gemini Embeddings** (using `text-embedding-004`) and supports fallback/swapping to **OpenAI** (`text-embedding-3-large`) and **Hugging Face** (`BAAI/bge-large-en-v1.5` via inference API).
   - Segregates data into two distinct collections: `api_docs` and `company_knowledge` for routing optimization.

3. **Hybrid Retrieval (`app/db.py`)**:
   - **Semantic Search**: Vector similarity query against the database using cosine distances.
   - **Keyword Search**: Okapi BM25 keyword lookup over tokenized document text corpus.
   - **Fusion**: Merges candidate sets from both methods using **Reciprocal Rank Fusion (RRF)** to return the top `k=5` highest relevance chunks globally.

4. **Query Router (`app/routing.py`)**:
   - Uses a high-performance query intent classifier. Queries are routed to the `api_docs` collection, the `company_knowledge` collection, or `mixed` (which retrieves from both collections simultaneously).
   - Built on an LLM classification prompt with a regular expression keyword-based fallback rule engine.

5. **Conversational Memory (`app/memory.py`)**:
   - Maintains a window of conversation history.
   - Implements **Query Contextualization**: Before executing search queries, the history and user question are passed to the LLM to rewrite ambiguous pronouns (e.g., "What parameters does it need?") into self-contained search terms (e.g., "What parameters does the WHOIS Lookup API require?").

6. **Prompt Engineering & Citations (`app/rag_engine.py`)**:
   - Instructs the model to answer **only** based on the provided context, return an explicit error string ("I could not find this information in the available documentation.") when text is unavailable, and cite source paths directly.
   - The RAG engine parses and extracts clean, unique metadata references from the matching chunks to return structured sources alongside the text.

7. **Streamlit UI (`app/ui.py`)**:
   - Features a high-fidelity glassmorphism dark theme withOutfit/Inter typography and custom CSS.
   - Features swappable side-panel configuration controls, real-time database chunk statistics, a manual ingestion button, and a clean chat trigger.
   - Renders interactive citations that can be expanded to view the matching document snippet.

8. **RAG Evaluation Suite (`app/evaluate.py`)**:
   - Programmatically audits the system using ground-truth test queries.
   - Calculates **Routing Accuracy**, **Faithfulness Rate** (hallucination detection via LLM auditor), **Response Relevance**, **Citation Match Rate**, and **Latency**.

---

## Directory Structure

```text
project/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configures paths, keys, token parameters, and defaults
│   ├── embeddings.py       # Modular interface for Gemini/OpenAI/HuggingFace embeddings
│   ├── llm.py              # Swappable model layer (Gemini, GPT, Claude)
│   ├── db.py               # ChromaDB interface, BM25 indices, and RRF Hybrid search
│   ├── ingestion.py        # Semantic block parser and metadata frontmatter chunker
│   ├── routing.py          # Classifies query intents (API, Knowledge, Mixed)
│   ├── memory.py           # Message history manager & LLM query contextualizer
│   ├── rag_engine.py       # System coordinator combining all RAG stages
│   ├── ui.py               # Streamlit application entrypoint
│   ├── evaluate.py         # Evaluation test harness and LLM auditing metrics
│   └── run_ingestion.py    # Command line ingestion trigger script
├── docs/                   # API reference markdown folders
├── knowledge/              # General website support text files
├── requirements.txt        # Python libraries pinned for virtual environments
├── run.py                  # Master project CLI entrypoint
└── .env                    # System secret keys (GEMINI_API_KEY, HF_TOKEN)
```

---

## Environment Setup

Verify that the `.env` file under `project/` contains your API tokens:

```env
GEMINI_API_KEY="your-gemini-api-key"
GROQ_API_KEY="your-groq-api-key"
HF_TOKEN="your-huggingface-token"
```

*Note: If `GROQ_API_KEY` is set, the system will automatically default to Groq's `llama-3.3-70b-versatile` for text generation tasks, which offers extremely high rate limits and is perfect for running the daily evaluation test suite.*

---

## Run Instructions

Use the master script `run.py` as your control panel:

### 1. Ingest Documents (Run Ingestion Pipeline)
Executes the document chunking, generates embeddings via Gemini, and indexes them in ChromaDB.
```bash
python run.py --ingest
```

### 2. Start Customer Support UI (Streamlit Frontend)
Launches the web application chat assistant.
```bash
python run.py --ui
# Or simply run without flags:
python run.py
```

### 3. Run Web Scraper (Optional)
If you need to refresh the source documents from the website again, run:
```bash
python run.py --scrape
```

---

## Example Queries to Try

### 📂 Company Knowledge Questions
* *What is WHOIS Data Center?* (Expected: Routing to `company_knowledge`, retrieved from `about_us.txt` or `homepage_overview.txt`)
* *What is their mission and vision?* (Expected: Routing to `company_knowledge`)
* *Why should I choose WHOIS Data Center over manual searches?* (Expected: Routing to `company_knowledge`)

### 🛠️ API Documentation Questions
* *How do I authenticate with the WHOIS API?* (Expected: Routing to `api_docs`, retrieves `authentication.md` showing Bearer Header tokens)
* *What parameters does the Historical WHOIS API require?* (Expected: Routing to `api_docs`, retrieves parameters table)
* *Show me a Python request example of the WHOIS Lookup API.* (Expected: Routing to `api_docs`, retrieves python code block from `whois_lookup.md`)

### 🔀 Mixed Context Questions
* *What services does WHOIS Data Center offer and how do I authenticate with their Reverse WHOIS API?* (Expected: Routing to `mixed`, queries both collections, answers both facts, and lists sources from both `about_us.txt` and `authentication.md`)

### 🧠 Conversational Memory Test
1. Ask: *What is WHOIS Lookup?*
2. Follow-up: *What parameters does it need?* (System will contextualize "it" to "WHOIS Lookup API" and correctly pull parameter details from `whois_lookup.md`)
