import pandas as pd

df = pd.read_csv(
    r"data\\processed\\cleaned_qa.csv",
    encoding="utf-8"
)

eval_df = pd.DataFrame({
    "question": df.loc[[0,1,2,3,18], "question"].values,
    "expected_row_id": [0,1,2,3,18]
})

eval_df.to_csv(
    r"data\\eval\\eval_questions_2.csv",
    index=False,
    encoding="utf-8-sig"
)

print(eval_df)