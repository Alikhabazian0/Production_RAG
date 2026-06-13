import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful assistant answering only from the provided context.
If the answer is not in the context, say you do not know.
Cite the source rows or URLs when available.
"""

def generate_answer(query: str, retrieved_docs):
    context = "\n\n---\n\n".join(
        [
            f"Source {doc.metadata}\nContent:\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    user_prompt = f"""
Question:
{query}

Context:
{context}
"""
    
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = f"""
    {SYSTEM_PROMPT}

    {user_prompt}
    """,
    config = {
        "temperature": 0.2
    }
    )

    return response.text

if __name__ == "__main__":
    from src.retriever import retrieve

    query = "شرکت‌های غیر دانش‌بنیان چه حمایت‌هایی می‌گیرند؟"

    docs = retrieve(query, k=3)

    answer = generate_answer(query, docs)

    print("\nQuestion:")
    print(query)

    print("\nGenerated answer:")
    print(answer)