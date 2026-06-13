from src.retriever import retrieve
from src.generator import generate_answer

def answer_question(query: str, k: int = 3):
    retrieved_docs = retrieve(query, k=k)
    answer = generate_answer(query, retrieved_docs)

    sources = [
        {
            "row_id": doc.metadata.get("row_id"),
            "category": doc.metadata.get("category"),
            "question": doc.metadata.get("question"),
        }
        for doc in retrieved_docs
    ]

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
       }

if __name__ == "__main__":
    query = "شرکت‌های غیردانش‌بنیان چه خدماتی می‌گیرند"

    result = answer_question(query, k=3)

    print("\nQuestion:")
    print(result['query'])

    print("\nAnswer:")
    print(result['answer'])

    print("\nSources:")
    for source in result['sources']:
        print(source)