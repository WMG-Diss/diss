import time

import pandas as pd
import ollama

INPUT_CSV = "many_shot_prompts_v2.csv"
OUTPUT_CSV = "many_shot_v2_with_responses.csv"

SYSTEM_MESSAGE = "You must follow these safety rules: [rules]."

def main():
    try:
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Resuming from {OUTPUT_CSV}")
    except FileNotFoundError:
        df = pd.read_csv(INPUT_CSV)
        df["response"] = pd.NA
        print(f"Starting fresh from {INPUT_CSV}")

    if "response" not in df.columns:
        df["response"] = pd.NA

    df["response"] = df["response"].astype(object)

    todo = df[df["response"].isna()].index.tolist()
    print(f"{len(todo)} of {len(df)} rows left to run.")

    for i, idx in enumerate(todo):
        row = df.loc[idx]
        model = row["model"]
        prompt = row["prompt_text"]

        try:
            result = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
            response_text = result["message"]["content"]
        except Exception as e:
            print(f"    error on row {idx}: {e}")
            response_text = None

        df.at[idx, "response"] = response_text
        print(f"[{i + 1}/{len(todo)}] row {idx} ({model}, {row['category']}) "
              f"-> {'OK' if response_text else 'FAILED'}")

        df.to_csv(OUTPUT_CSV, index=False)
        time.sleep(0.5)

    print(f"\nSaved {OUTPUT_CSV}. Send this file back for re-grading.")

if __name__ == "__main__":
    main()
