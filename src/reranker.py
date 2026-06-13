from sentence_transformers import CrossEncoder

from src.hybrid_retriever import hybrid_retrieve

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

reranker = CrossEncoder(RERANKER_MODEL)

def rerank_documents(query: str, docs, top_k: int=3):
    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse = True
    )

    return [doc for doc, score in ranked[:top_k]]

def retrieve_with_reranking(query: str, candidate_k: int = 10, final_k: int = 3):
    candidate_docs = hybrid_retrieve(
        query=query,
        k=candidate_k,
        vector_k = candidate_k,
        bm25_k = candidate_k
    )

    final_docs = rerank_documents(
        query=query,
        docs = candidate_docs,
        top_k=final_k
    )

    return final_docs

if __name__ == "__main__":
    query = "سقف تسهیلات دانش بنیان چقدر است؟"

    results = retrieve_with_reranking(
        query=query,
        candidate_k=10,
        final_k=3
    )

    for i, doc in enumerate(results, start=1):
        print("=" * 50)
        print(f"Result {i}")
        print(doc.metadata)
        print(doc.page_content[:700])