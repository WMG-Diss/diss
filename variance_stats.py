import pandas as pd
import numpy as np

INPUT_CSV = "variance_graded.csv"

def compute_stability(df):
    grouped = df.groupby("_orig_row_id")
    results = []
    for row_id, g in grouped:
        labels = g["llm_label"].tolist()
        broad = [1 if l >= 1 else 0 for l in labels]
        results.append(
            {
                "row_id": row_id,
                "category": g["category"].iloc[0],
                "pattern": g["pattern"].iloc[0],
                "labels": labels,
                "exact_stable": len(set(labels)) == 1,
                "broad_stable": len(set(broad)) == 1,
                "mean_broad": np.mean(broad),
                "full_swing": 0 in labels and 2 in labels,
            }
        )
    return pd.DataFrame(results)

def summarise(res):
    n = len(res)
    return {
        "n_prompts": n,
        "exact_stable_n": int(res["exact_stable"].sum()),
        "exact_stable_pct": round(res["exact_stable"].mean() * 100, 1),
        "broad_stable_n": int(res["broad_stable"].sum()),
        "broad_stable_pct": round(res["broad_stable"].mean() * 100, 1),
        "full_swing_n": int(res["full_swing"].sum()),
        "full_swing_pct": round(res["full_swing"].mean() * 100, 1),
    }

def by_pattern(res):
    return res.groupby("pattern")["exact_stable"].sum()

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    res = compute_stability(df)
    print(summarise(res))
    print(by_pattern(res))
