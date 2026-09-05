# PDF RAG Chatbot (From First Principles)

## Project Goal

This project implements a Retrieval-Augmented Generation (RAG) system completely from scratch without using LangChain, ChromaDB, FAISS, Pinecone, or any other high-level framework.

The objective was to understand every component involved in a RAG pipeline before relying on abstractions.

The project was built as a learning exercise covering:

- PDF processing
- Text chunking
- Embeddings
- Semantic search
- Vector retrieval
- Prompt construction
- Local LLM inference


The final system can answer questions about a PDF document by retrieving relevant context and passing it to a local LLM running through Ollama.

---

# What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

Instead of asking an LLM to answer only from its training data:

```text
Question
    ↓
   LLM
    ↓
 Answer
```

RAG first retrieves relevant information from an external knowledge source:

```text
Question
    ↓
Retriever
    ↓
Relevant Chunks
    ↓
   LLM
    ↓
 Answer
```

This allows the model to answer questions about documents it was never trained on.

---

# Architecture

## Offline Phase (Indexing)

This phase runs once.

```text
PDF
 ↓
Load PDF
 ↓
Extract Text
 ↓
Chunk Text
 ↓
Generate Embeddings
 ↓
Store Index
```

Output:

```text
indexes/index.pkl
```

---

## Online Phase (Question Answering)

This phase runs for every user query.

```text
Question
 ↓
Embed Query
 ↓
Similarity Search
 ↓
Retrieve Top-K Chunks
 ↓
Build Prompt
 ↓
Llama 3.2 (Ollama)
 ↓
Answer
```

---

# Project Structure

```text
pdf-rag-chatbot/

├── data/
│   └── Attention_Is_All_You_Need.pdf

├── indexes/
│   └── index.pkl

├── src/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── indexer.py
│   ├── prompt_builder.py
│   ├── llm.py
│   └── search.py

├── requirements.txt
├── README.md
└── .gitignore
```

---

# Components

## pdf_loader.py

Responsible for reading PDF files.

Uses:

```python
from pypdf import PdfReader
```

Output:

```python
[
    page1_text,
    page2_text,
    ...
]
```

Purpose:

Convert PDF content into raw text.

---

## chunker.py

Responsible for splitting large text into smaller overlapping chunks.

Current settings:

```python
chunk_size = 500
overlap = 100
```

Result:

```text
Chunk 1: 0-500
Chunk 2: 400-900
Chunk 3: 800-1300
...
```

Purpose:

- Fit within LLM context limits
- Preserve semantic continuity across chunk boundaries

---

## indexer.py

Builds the searchable index.

Pipeline:

```text
PDF
 ↓
Text
 ↓
Chunks
 ↓
Embeddings
 ↓
index.pkl
```

Purpose:

Precompute embeddings so they do not need to be regenerated for every query.

---

## retriever.py

Performs semantic search.

Pipeline:

```text
Question
 ↓
Query Embedding
 ↓
Cosine Similarity
 ↓
Ranking
 ↓
Top-K Results
```

Purpose:

Find the most relevant chunks for a query.

---

## prompt_builder.py

Constructs the prompt sent to the LLM.

Combines:

```text
Question
+
Retrieved Chunks
```

Purpose:

Provide the model with relevant context before generation.

---

## llm.py

Communicates with Ollama.

Current model:

```text
llama3.2
```

Purpose:

Generate answers from retrieved context.

---

## search.py

Main application entry point.

Pipeline:

```text
Load Index
 ↓
Retrieve Chunks
 ↓
Build Prompt
 ↓
Call LLM
 ↓
Display Answer
```

Purpose:

Orchestrates the complete RAG workflow.

---

# Index Structure

The project stores a precomputed index inside:

```text
indexes/index.pkl
```

Structure:

```python
{
    "chunks": [
        {
            "text": "...",
            "source": "data/Attention_Is_All_You_Need.pdf"
        }
    ],

    "embeddings": [...]
}
```

The index contains:

- Chunk text
- Metadata
- Embeddings

Purpose:

Avoid recomputing embeddings during every search.

---

# Retrieval Logic

Current retrieval strategy:

```text
1. Embed the user query
2. Compare against all chunk embeddings
3. Compute cosine similarity
4. Rank by similarity
5. Return Top-K chunks
```

Current implementation:

```text
Brute Force Search
```

Every query is compared against every stored embedding.

Complexity:

```text
Similarity Search: O(n)

Ranking:
O(n log n)
```

This works well for small datasets but becomes expensive at scale.

Vector databases and FAISS solve this problem using efficient nearest-neighbor search.

---

# Embeddings

Model used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Purpose:

Convert text into vectors that capture semantic meaning.

Example:

```text
Transformers use self attention

≈

Attention mechanisms are used in transformers
```

Similar meanings produce nearby vectors.

---

# Cosine Similarity

Cosine similarity measures the angle between vectors.

Purpose:

Determine semantic similarity.

Example:

```text
Transformer sentence
      vs
Transformer sentence
```

High similarity score.

```text
Transformer sentence
      vs
Cats are mammals
```

Low similarity score.

Reason for using cosine similarity:

It measures direction rather than magnitude.

---

# Prompt Design

Prompt structure:

```text
Question
+
Retrieved Chunks
```

Current rules:

- Use only provided context
- Do not invent information
- If information is unavailable, respond:

"I don't know based on the provided document."
```

Purpose:

Reduce hallucinations and force grounding in retrieved content.

---

# Metadata

Each chunk stores metadata:

```python
{
    "text": "...",
    "source": "..."
}
```

Purpose:

Track where retrieved information originated.

This introduces a key vector database concept:

```text
Embedding
+
Document
+
Metadata
```

---

# Key Insights

### RAG is fundamentally:

```text
Chunks
+
Embeddings
+
Retrieval
+
Prompt
+
LLM
```

Everything else is an abstraction built on top of these components.

---

### Vector Databases Are Not Magic

A vector database fundamentally stores:

```text
Embedding
+
Document
+
Metadata
```

and provides efficient nearest-neighbor search.

---

### LangChain Is Not RAG

LangChain is an orchestration framework.

RAG can be implemented entirely without LangChain.

---

### Retrieval Quality Matters

A powerful LLM cannot answer correctly if retrieval fails.

Pipeline:

```text
Bad Retrieval
 ↓
Bad Context
 ↓
Bad Answer
```

Improving retrieval is often more valuable than switching to a larger model.

---

### Indexing And Retrieval Are Separate Problems

Indexing:

```text
PDF
 ↓
Embeddings
 ↓
Storage
```

Retrieval:

```text
Question
 ↓
Similarity Search
 ↓
Relevant Chunks
```

This separation is the foundation of production RAG systems.

---

# Running The Project

Build the index:

```bash
python src/indexer.py
```

Start the chatbot:

```bash
python src/search.py
```

Exit:

```text
exit
```

---

# Future Improvements

Potential next steps:

- Replace brute-force retrieval with FAISS
- Replace pickle storage with ChromaDB
- Add page-level citations
- Support multiple PDFs
- Add reranking models
- Add hybrid search (keyword + semantic)
- Compare manual pipeline with LangChain
- Experiment with larger local models
- Add conversational memory
- Build a web interface

---

# Final Takeaway

This project demonstrates that a RAG system is not magic.

At its core:

```text
PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
Retrieval
 ↓
Prompt
 ↓
LLM
 ↓
Answer
```

Understanding these fundamentals makes frameworks such as LangChain, ChromaDB, FAISS, Pinecone, Qdrant, and Weaviate much easier to learn because they are abstractions over the same underlying ideas.