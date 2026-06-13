from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for doc in docs:
        if len(doc.page_content) <= CHUNK_SIZE:
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))

    return chunks

if __name__ == "__main__":
    from src.ingest import load_qa_documents

    docs = load_qa_documents()
    chunks = chunk_documents(docs)

    print(f"Original document: {len(docs)}")
    print(f"total chunks: {len(chunks)}")

    print("\nFirst chunk:")
    print(chunks[0].page_content)

    print("\nFirst chunk metadata:")
    print(chunks[0].metadata)