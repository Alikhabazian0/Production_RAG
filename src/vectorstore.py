from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vectorstore(chunks):
    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        collection_name = COLLECTION_NAME,
        persist_directory = str(CHROMA_DIR)
    ) 

    return vectorstore

# def load_vectorstore():
#     embeddings = get_embedding_model()

#     return Chroma(
#         collection_name = COLLECTION_NAME,
#         embedding_function = embeddings,
#         persist_directory = str(CHROMA_DIR)
#     )

### fixing repeated model loading with caching/singletons
_VECTORSTORE = None

def load_vectorstore():
    global _VECTORSTORE

    if _VECTORSTORE is None:
        embeddings = get_embedding_model()

        _VECTORSTORE = Chroma(
            collection_name = COLLECTION_NAME,
            embedding_function = embeddings,
            persist_directory = str(CHROMA_DIR)
        )

    return _VECTORSTORE

if __name__ == "__main__":
    from src.ingest import load_qa_documents
    from src.chunk import chunk_documents

    docs = load_qa_documents()
    chunks = chunk_documents(docs)

    print(f"Building vector DB from {len(chunks)} chunks...")

    vectorstore = build_vectorstore(chunks)

    print("Vector DB build successfully.")

    results = vectorstore.similarity_search(
        "شرکت‌های غیردانش‌بنیان از چه خدماتی می‌توانند استفاده کنند؟",
        k=3
    )

    print("\ntop restricted result:")
    print(results[0].page_content)
    print(results[0].metadata)