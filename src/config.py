from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_CSV = BASE_DIR / "data/raw/qa_crawled.csv"
CLEAN_CSV = BASE_DIR / "data/processed/cleaned_qa.csv"
CHROMA_DIR = BASE_DIR / "vector_db/chroma"

# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  -- old version
EMBEDDING_MODEL =  "BAAI/bge-m3"

COLLECTION_NAME = "persian_qa_rag"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
TOP_K = 5