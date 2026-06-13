from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_pipeline import answer_question

app = FastAPI(
    title = "Persian RAG API",
    vesion = "1.0.0"
)

class QuestionRequest(BaseModel):
    question: str
    k: int = 3


@app.get("/")
def root():
    return {"message": "Persian RAG API is running."}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = answer_question(
        query = request.question,
        k = request.k
    )
    return result
