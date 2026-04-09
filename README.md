# 📚 RAG Book Assistant

A Retrieval-Augmented Generation (RAG) application that lets you upload PDF documents and ask natural language questions — getting answers grounded strictly in your document's content.

Two implementations are available: one powered by **Mistral AI** and another by **Google Gemini**, both sharing the same core RAG architecture.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [Step 1 — PDF Ingestion](#step-1--pdf-ingestion)
  - [Step 2 — Text Chunking](#step-2--text-chunking)
  - [Step 3 — Embedding Generation](#step-3--embedding-generation)
  - [Step 4 — Vector Storage](#step-4--vector-storage)
  - [Step 5 — Query & Retrieval](#step-5--query--retrieval)
  - [Step 6 — Answer Generation](#step-6--answer-generation)
- [Architecture Diagram](#architecture-diagram)
- [Implementations](#implementations)
  - [app.py — Mistral AI + ChromaDB](#apppy--mistral-ai--chromadb)
  - [main.py — Google Gemini + FAISS](#mainpy--google-gemini--faiss)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Design Decisions](#design-decisions)

---

## Overview

Traditional LLMs answer from their training data alone — they hallucinate when asked about documents they've never seen. This project solves that by implementing **RAG (Retrieval-Augmented Generation)**: the model is forced to answer only from content retrieved from your uploaded PDF, making responses accurate and document-grounded.

---

## How It Works

### Step 1 — PDF Ingestion

The uploaded PDF file is read and its raw text is extracted page by page. This gives the pipeline the full textual content of the document to work with.

### Step 2 — Text Chunking

The extracted text is too large to fit into a single LLM context window, so it is split into smaller overlapping chunks using a **Recursive Character Text Splitter**. Each chunk is 1,000 characters long with a 200-character overlap between consecutive chunks. The overlap ensures that context is not lost at chunk boundaries.

### Step 3 — Embedding Generation

Each text chunk is converted into a numerical vector (an "embedding") using a **HuggingFace Sentence Transformer** model (`all-MiniLM-L6-v2`). Embeddings capture the semantic meaning of text — chunks with similar meanings end up close together in vector space, enabling similarity-based search later.

### Step 4 — Vector Storage

All the generated embeddings are stored in a **vector database** so they can be searched efficiently at query time. This project supports two vector stores:

- **ChromaDB** — a persistent local vector database (used in `app.py`)
- **FAISS** — Facebook AI Similarity Search, a high-performance in-memory index saved to disk (used in `main.py`)

Both databases allow you to search millions of vectors by semantic similarity in milliseconds.

### Step 5 — Query & Retrieval

When the user submits a question, it is converted into an embedding using the same HuggingFace model. The vector database then searches for the most semantically similar chunks from the document — these are the chunks most likely to contain the answer.

- `app.py` uses **MMR (Maximal Marginal Relevance)** retrieval, which balances relevance and diversity to avoid returning repetitive chunks.
- `main.py` uses standard **similarity search**.

The top retrieved chunks are assembled into a **context block**.

### Step 6 — Answer Generation

The retrieved context, along with the user's question, is formatted into a structured **prompt** and sent to the LLM. The prompt explicitly instructs the model to answer only from the provided context — if the answer isn't there, the model says so rather than guessing.

The LLM generates the final answer, which is displayed to the user in the Streamlit interface.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        INDEXING PHASE                        │
│                  (runs once per document)                    │
│                                                             │
│  PDF Upload ──► Text Extraction ──► Chunking ──► Embedding  │
│                                                      │       │
│                                               Vector Store   │
│                                          (ChromaDB / FAISS)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        QUERY PHASE                           │
│                   (runs on every question)                   │
│                                                             │
│  User Question ──► Embed Question ──► Vector Search          │
│                                            │                 │
│                                   Top-K Relevant Chunks      │
│                                            │                 │
│                              Prompt = Context + Question      │
│                                            │                 │
│                                    LLM (Mistral / Gemini)    │
│                                            │                 │
│                                      Final Answer            │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementations

### `app.py` — Mistral AI + ChromaDB

| Component | Choice |
|---|---|
| PDF Loader | LangChain `PyPDFLoader` |
| Vector Store | ChromaDB (persistent on disk) |
| Retrieval | MMR (diversity-aware) |
| LLM | Mistral AI (`mistral-small-2506`) |
| Framework | LangChain + Streamlit |

**Flow:** The user uploads a PDF and clicks "Create Vector Database." The app processes the document and saves the ChromaDB index to a local folder. On every subsequent run, if the index exists, the app loads it directly — no reprocessing needed. Questions are answered via Mistral AI using a strict system prompt.

---

### `main.py` — Google Gemini + FAISS

| Component | Choice |
|---|---|
| PDF Loader | `PyPDF2` |
| Vector Store | FAISS (saved to disk as `faiss_index/`) |
| Retrieval | Standard similarity search |
| LLM | Google Gemini (`gemini-1.5-flash`) |
| Framework | LangChain + Streamlit |

**Flow:** PDFs are uploaded via the sidebar. Clicking "Process PDF" extracts text, chunks it, generates embeddings, and saves the FAISS index. Questions are typed in the main area and answered via the Gemini API using a detailed prompt template.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDF2, PyPDFLoader (LangChain) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB / FAISS |
| LLM (Option A) | Mistral AI via `langchain-mistralai` |
| LLM (Option B) | Google Gemini via `google-generativeai` |
| Prompt Management | LangChain `ChatPromptTemplate` / `PromptTemplate` |
| Environment Config | `python-dotenv` |

---

## Setup & Installation

**1. Clone the repository and navigate to the project folder.**

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install all dependencies:**

```bash
pip install -r requirements.txt
```

**4. Configure your API keys** (see section below).

**5. Run the app of your choice:**

```bash
# Mistral AI version
streamlit run app.py

# Google Gemini version
streamlit run main.py
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# For app.py (Mistral AI)
MISTRAL_API_KEY=your_mistral_api_key_here

# For main.py (Google Gemini)
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Note:** Never commit your `.env` file to version control. Add it to `.gitignore`.

---

## Usage

1. Launch the app with `streamlit run app.py` (or `main.py`).
2. Upload a PDF document using the file uploader.
3. Click the button to process the document and build the vector index.
4. Type your question in the text input.
5. The app retrieves the most relevant sections from your document and generates a grounded answer.

If the answer is not present in the document, the model will explicitly say so — it will not hallucinate an answer.

---

## Design Decisions

**Why RAG instead of just sending the whole PDF to the LLM?**
LLMs have context window limits. A 300-page book can't fit in a single prompt. RAG selectively retrieves only the relevant sections, making it scalable to documents of any size.

**Why local HuggingFace embeddings instead of OpenAI embeddings?**
HuggingFace `all-MiniLM-L6-v2` is free, fast, runs locally, and produces high-quality embeddings for most document Q&A use cases — no additional API cost.

**Why chunk overlap?**
Without overlap, answers that span a chunk boundary would be missed. A 200-character overlap ensures continuity between adjacent chunks.

**Why MMR retrieval in `app.py`?**
Standard similarity search can return multiple near-duplicate chunks. MMR (Maximal Marginal Relevance) balances relevance with diversity, ensuring the retrieved context covers more ground.
