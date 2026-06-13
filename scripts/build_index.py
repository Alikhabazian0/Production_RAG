from src.clean import clean_csv
from src.ingest import load_qa_documents
from src.chunk import chunk_documents
from src.vectorstore import build_vectorstore

def main():
    clean_csv()
    docs = load_qa_documents()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)

    print(f"document: {len(docs)}")
    print(f"chunks: {len(chunks)}")
    print("vector index built successfully.")

if __name__ == "__main__":
    main()