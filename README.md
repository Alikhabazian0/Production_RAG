# Persian RAG Question Answering System

## Overview

This project implements a production-style Retrieval-Augmented Generation (RAG) system for Persian Question Answering.

The system retrieves relevant documents from a Persian knowledge base using semantic search and generates grounded answers using Google's Gemini API.

The project was developed as an end-to-end AI Engineering portfolio project covering:

* Data preprocessing
* Document chunking
* Embedding generation
* Vector database creation
* Retrieval evaluation
* Embedding benchmarking
* FastAPI backend development
* Streamlit frontend development
* Docker deployment

---

# System Architecture

```text
User Question
      │
      ▼
Streamlit Frontend
      │
      ▼
FastAPI Backend
      │
      ▼
Retriever (BGE-M3)
      │
      ▼
Chroma Vector Database
      │
      ▼
Top-K Relevant Documents
      │
      ▼
Gemini 2.5 Flash
      │
      ▼
Grounded Answer
      │
      ▼
Response + Sources
```

---

# Project Structure

```text
Production_RAG/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── qa_crawled.csv
│   │
│   ├── processed/
│   │   └── cleaned_qa.csv
│   │
│   └── eval/
│       └── eval_questions.csv
│
├── reports/
│   ├── minilm_evaluation.txt
│   └── bge_m3_evaluation.txt
│
├── scripts/
│
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── clean.py
│   ├── chunk.py
│   ├── embed.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── hybrid_retriever.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   └── evaluation.py
│
├── vector_db/
│
├── tests/
│
├── notebooks/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Dataset

The dataset consists of Persian question-answer pairs collected from a public knowledge source.

Dataset processing pipeline:

1. Raw data ingestion
2. Data cleaning
3. Text normalization
4. Chunk creation
5. Embedding generation
6. Vector database construction

---

# Embedding Experiments

## Baseline Model

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Results:

```text
HitRate@3 = 85%
HitRate@5 = 90%
```

Observed issues:

* Struggled with paraphrased questions
* Confused semantically related financial concepts
* Reduced retrieval quality for unseen wording

---

## Reranker Experiment

Model tested:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Observation:

```text
No significant improvement in retrieval performance.
Additional latency introduced.
Not selected for production.
```

---

## Final Production Model

```text
BAAI/bge-m3
```

Results:

```text
HitRate@3 = 100%
HitRate@5 = 100%
```

Benefits:

* Strong multilingual support
* Better semantic understanding
* Superior retrieval quality on Persian queries
* Better handling of paraphrases

---

# Evaluation Methodology

Evaluation was performed using a manually curated benchmark dataset containing Persian questions and expected document identifiers.

Example:

```csv
question,expected_row_id
شرکت دانش بنیان چیست؟,2
```

Evaluation metric:

```text
HitRate@K
```

Definition:

A query is counted as successful if the expected document appears within the top K retrieved documents.

---

# FastAPI Backend

Main endpoint:

```http
POST /ask
```

Request:

```json
{
  "question": "شرکت دانش بنیان چیست؟",
  "k": 3
}
```

Response:

```json
{
  "query": "...",
  "answer": "...",
  "sources": [...]
}
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

# Streamlit Frontend

The Streamlit application provides:

* Interactive question answering
* Configurable retrieval depth
* Source inspection
* End-user interface

Run:

```bash
streamlit run app/streamlit_app.py
```

---

# Local Installation

Clone repository:

```bash
git clone <repository-url>
cd Production_RAG
```

Create environment:

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment file:

```bash
cp .env.example .env
```

Add Gemini API key:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

# Build Vector Database

```bash
python -m src.vectorstore
```

---

# Run FastAPI

```bash
uvicorn api.main:app --reload
```

---

# Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

# Docker Deployment

Build and run:

```bash
docker compose up --build
```

Services:

```text
FastAPI   : http://localhost:8000
Swagger   : http://localhost:8000/docs
Streamlit : http://localhost:8501
```

---

# Engineering Lessons Learned

This project highlighted several practical AI engineering challenges:

* Embedding model selection strongly affects retrieval quality
* Evaluation datasets are essential for measuring progress
* Larger embedding models can significantly improve retrieval performance
* Rerankers do not always provide measurable gains
* Separating frontend and backend simplifies deployment
* Retrieval quality should be measured before changing LLMs
* Dockerization improves portability and reproducibility

---

# Future Improvements

Potential future extensions:

* Hybrid Retrieval (Dense + BM25)
* Query Expansion
* Parent-Child Retrieval
* Multi-query Retrieval
* Better Persian reranking models
* LangSmith monitoring
* Kubernetes deployment
* Authentication and user management
* Streaming responses
* Citation highlighting

---

# Author

Developed as an end-to-end Persian RAG and AI Engineering project demonstrating:

* Retrieval-Augmented Generation
* Semantic Search
* Evaluation-Driven Development
* FastAPI
* Streamlit
* Docker
* Production-Oriented AI Engineering

```
```
