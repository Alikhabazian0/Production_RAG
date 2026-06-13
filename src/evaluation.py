import pandas as pd
from pathlib import Path

from src.retriever import retrieve 

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_PATH = BASE_DIR / "data/eval/eval_questions_hard.csv" # change base on the subject

def evaluate_retrieval(k: int = 3):
    df = pd.read_csv(EVAL_PATH)

    total = len(df)
    hits = 0

    for _, row in df.iterrows():
        question = row["question"]
        expected_row_id = int(row['expected_row_id'])

        docs = retrieve(question, k=k)
        retrieved_ids = [doc.metadata.get("row_id") for doc in docs]

        is_hit = expected_row_id in retrieved_ids

        if is_hit:
            hits += 1

        print("=" * 50)
        print(f"Question: {question}")
        print(f"Expected row_id: {expected_row_id}")
        print(f"Retrieved row_ids: {retrieved_ids}")
        print(f"hit: {is_hit}")

    hit_rate = hits / total

    print("\nFinal Evaluation")
    print(f"Total questions: {total}")
    print(f"Hits: {hits}")
    print(f"Hit Rate@{k}: {hit_rate:.2%}")

if __name__ == "__main__":
    evaluate_retrieval(k=5)