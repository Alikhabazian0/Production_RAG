import pandas as pd
from langchain_core.documents import Document
from src.config import CLEAN_CSV

# Document has:
# page_content = text used for search
# metadata = extra information like category, row_id


def load_qa_documents():
    df = pd.read_csv(CLEAN_CSV)

    docs = []
    for i, row in df.iterrows():
        content = f"""

دسته بندی:
{row["category"]}

سوال:
{row['question']}

پاسخ:
{row['answer']}
""".strip()
        # content = f"""... Creates the searchable text.
        # This is what gets embedded into the vector database.
        metadata = {
            "row_id": int(i),
            "category": row['category'],
            'question': row['question']
        }

        docs.append(Document(
            page_content=content, 
            metadata=metadata))

    return docs

if __name__ == "__main__":
    docs = load_qa_documents()

    print(f"loaded documents: {len(docs)}")
    print("\nFirst document content:")
    print(docs[0].page_content)

    print("\nFirst document metadata:")
    print(docs[0].metadata)