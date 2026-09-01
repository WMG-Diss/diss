import argparse

import pandas as pd

INPUT_CSV = "results_log_graded.csv"
OUTPUT_CSV = "stratified_sample_for_hand_check.csv"
ANSWER_KEY_CSV = "stratified_sample_answer_key.csv"
TARGET_TOTAL = 120

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--answer-key", default=ANSWER_KEY_CSV)
    parser.add_argument("--target", type=int, default=TARGET_TOTAL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["llm_label"].notna()].copy()
    if df.empty:
        raise SystemExit(
            "No graded rows found yet -- run grade_responses.py first "
            "(it doesn't need to be finished, but needs at least some "
            "graded rows to sample from)."
        )
    df["_row_id"] = df.index

    strata = df.groupby(["model", "category"], dropna=False)
    n_strata = strata.ngroups
    per_stratum = max(1, round(args.target / n_strata))

    print(f"{n_strata} strata (model x category), "
          f"~{per_stratum} rows each -> target ~{per_stratum * n_strata}")

    sampled = strata.apply(
        lambda g: g.sample(n=min(per_stratum, len(g)), random_state=args.seed),
        include_groups=False,
    )
    sampled = sampled.reset_index()
    drop_cols = [c for c in sampled.columns if c.startswith("level_")]
    sampled = sampled.drop(columns=drop_cols, errors="ignore")

    answer_key = sampled[["_row_id", "llm_label", "llm_rationale"]].copy()
    answer_key.to_csv(args.answer_key, index=False)

    blind = sampled.drop(columns=["llm_label", "llm_rationale",
                                   "suggested_label", "rationale", "label"],
                          errors="ignore")
    blind["human_label"] = ""
    blind["human_notes"] = ""
    cols = [c for c in blind.columns if c not in ("human_label", "human_notes")]
    blind = blind[cols + ["human_label", "human_notes"]]
    blind.to_csv(args.output, index=False)

    print(f"Saved {len(blind)} rows to {args.output} (blind -- work from this one)")
    print(f"Saved matching answers to {args.answer_key} (don't open until done)")
    print("\nBreakdown:")
    print(sampled.groupby(["model", "category"]).size())
    print(f"\nOpen {args.output} in Excel/Sheets, read each prompt+response, "
          f"and fill in human_label (0/1/2) for every row. Then run "
          f"compute_kappa.py.")

if __name__ == "__main__":
    main()
