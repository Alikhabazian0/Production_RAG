import pandas as pd
from rank_bm25 import BM25Okapi

from src.config import CLEAN_CSV
from src.retriever import retrieve
from src.ingest import load_qa_documents

_BM25 = None
_BM25_DOCS = None

def simple_persian_tokenize(text: str):
    return str(text).replace("", " ").split()


# def bm25_retrieve(query: str, k: int = 3):
#     docs = load_qa_documents()

#     tokenized_docs = [
#         simple_persian_tokenize(doc.page_content)
#         for doc in docs
#     ]

#     bm25 = BM25Okapi(tokenized_docs)

#     tokenized_query = simple_persian_tokenize(query)
#     scores = bm25.get_scores(tokenized_query)

#     top_indices = scores.argsort()[::-1][:k]

#     return [docs[i] for i in top_indices]

#### improving bm25_retrieve using globals

def get_bm25_index():
    global _BM25, _BM25_DOCS

    if _BM25 is None:
        _BM25_DOCS = load_qa_documents()

        tokenized_docs = [
            simple_persian_tokenize(doc.page_content)
            for doc in _BM25_DOCS
        ]

        _BM25 = BM25Okapi(tokenized_docs)

    return _BM25, _BM25_DOCS

def bm25_retrieve(query: str, k: int = 3):
    bm25, docs = get_bm25_index()

    tokenized_query = simple_persian_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[::-1][:k]

    return [doc[i] for i in top_indices]


# hybrid retrieval
def hybrid_retrieve(query: str, k: int = 3, vector_k: int = 5, bm25_k: int = 5):
    vector_docs = retrieve(query, k=vector_k)
    bm25_docs = bm25_retrieve(query, k=bm25_k)

    merged = {}

    for rank, doc in enumerate(vector_docs):
        row_id = doc.metadata.get('row_id')
        merged[row_id] = {
            "doc": doc,
            "score": 1 / (rank + 1)
        }

    for rank, doc in enumerate(bm25_docs):
        row_id = doc.metadata.get("row_id")
        if row_id not in merged:
            merged[row_id] = {
                "doc": doc,
                "score": 0
            }
        merged[row_id]['score'] += 1 / (rank + 1)

    ranked = sorted(
        merged.values(),
        key = lambda x: x['score'],
        reverse = True
    )

    return [item['doc'] for item in ranked[:k]]

if __name__ == "__main__":
    query = "سقف تسهیلات دانش بنیان چقدر است؟"

    results = hybrid_retrieve(query, k=3)

    for i, doc in enumerate(results, start=1):
        print("=" * 50)
        print(f"Result {i}")
        print(doc.metadata)
        print(doc.page_content[:700])