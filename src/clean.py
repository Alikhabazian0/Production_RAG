import pandas as pd
import re
from src.config import RAW_CSV, CLEAN_CSV

def normalize_persian_text(text: str) -> str:
    text = str(text)

    # arabic to persian character normalixation
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")

    # remove extra whitespaces
    # Replaces multiple spaces/newlines/tabs with one space.
    text = re.sub(r"\s+"," ", text)
    
    # Removes spaces from beginning and end
    return text.strip()

def clean_csv():
    df = pd.read_csv(RAW_CSV)

    expected_cols = {'category', 'question', 'answer'}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Mising columns: {missing}")
    
    # Delete rows where category, question, or answer is empty
    df = df.dropna(subset=['category', 'question', 'answer'])

    for col in ['category', 'question', 'answer']:
        df[col] = df[col].apply(normalize_persian_text)

    #Remove repeated "پاسخ:" from answer beginning
    df['answer'] = df['answer'].str.replace(r"^پاسخ\s*[::]\s*", "", regex=True)

    df = df[df['question'].str.len() > 5]
    df = df[df['answer'].str.len() > 10]
    
    df = df.drop_duplicates(subset=['category','question', 'answer'])

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False, encoding='utf-8-sig')
    ## utf-8-sig helps Persian text display correctly in Excel.
    
    print(f"Saved cleaned data: {len(df)} rows")

if __name__ == "__main__":
    clean_csv()