import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

INPUT_CSV = "stratified_sample_for_hand_check.csv"
ANSWER_KEY_CSV = "stratified_sample_answer_key.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--answer-key", default=ANSWER_KEY_CSV)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    answers = pd.read_csv(args.answer_key)
    df = df.merge(answers, on="_row_id", how="left")

    df["human_label"] = pd.to_numeric(df["human_label"], errors="coerce")
    missing = df["human_label"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows still have no human_label -- "
              f"excluding them from this calculation. Fill them in for "
              f"the real final number.")
    df = df.dropna(subset=["human_label"])

    if df.empty:
        raise SystemExit("No rows have a human_label yet -- nothing to compute.")

    kappa = cohen_kappa_score(df["human_label"], df["llm_label"])
    agreement_rate = (df["human_label"] == df["llm_label"]).mean()

    print(f"n = {len(df)} hand-checked rows")
    print(f"Raw agreement rate: {agreement_rate:.1%}")
    print(f"Cohen's kappa: {kappa:.3f}")
    print()
    print("Confusion matrix (rows = your label, columns = LLM label, "
          "labels 0/1/2):")
    cm = confusion_matrix(df["human_label"], df["llm_label"], labels=[0, 1, 2])
    print(pd.DataFrame(cm, index=["human=0", "human=1", "human=2"],
                        columns=["llm=0", "llm=1", "llm=2"]))
    print()
    print("Rows where you and the LLM disagreed (worth a quick look before "
          "reporting kappa -- these are your best evidence for the write-up "
          "either way):")
    disagreements = df[df["human_label"] != df["llm_label"]]
    if disagreements.empty:
        print("  (none -- perfect agreement)")
    else:
        cols = [c for c in ["model", "category", "pattern",
                             "human_label", "llm_label"] if c in df.columns]
        print(disagreements[cols].to_string(index=False))

if __name__ == "__main__":
    main()
