from src.vectorstore import load_vectorstore
from src.config import TOP_K

def retrieve(query: str, k: int = TOP_K):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return docs

if __name__ == "__main__":
    query = "شرکت‌های غیر دانش بنیان چه حمایت‌هایی می‌گیرند"

    results = retrieve(query, k=3)

    print(f"Query: {query}")
    print(f"Retrieved documents: {len(results)}")

    for i, doc in enumerate(results, start=1):
        print("\n" + "=" * 50)
        print(f"Result: {i}")
        print("Metadata: ", doc.metadata)
        print("Content:")
        print(doc.page_content[:1000])