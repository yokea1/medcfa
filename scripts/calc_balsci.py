import pandas as pd

# 分类指标
cls = pd.read_csv(
    "results/chestxdet10_classification_summary_with_negative.csv"
)

# flip指标
flip = pd.read_csv(
    "results/chestxdet10_flip_all_summary.csv"
)

# 只保留 matched_patch
flip = flip[flip["operator"] == "matched_patch"]

rows = []

for _, c in cls.iterrows():

    model = c["model_key"]

    f = flip[flip["model_key"] == model].iloc[0]

    recall = c["recall"]
    spec = c["specificity"]
    f1 = c["f1"]

    dflip = f["delta_flip"]

    balacc = (recall + spec) / 2

    balsci = balacc - dflip
    f1sci = f1 - dflip

    rows.append({
        "model": c["model_name"],
        "Recall": round(recall,4),
        "Specificity": round(spec,4),
        "F1": round(f1,4),
        "MP_DeltaFlip": round(dflip,4),
        "BalAcc": round(balacc,4),
        "BalSCI": round(balsci,4),
        "F1SCI": round(f1sci,4),
    })

out = pd.DataFrame(rows)

print("\n=== BalSCI Ranking ===")
print(
    out.sort_values(
        "BalSCI",
        ascending=False
    )[
        ["model","BalSCI"]
    ]
)

print("\n=== F1SCI Ranking ===")
print(
    out.sort_values(
        "F1SCI",
        ascending=False
    )[
        ["model","F1SCI"]
    ]
)

out.to_csv(
    "results/chestxdet10_balsci_summary.csv",
    index=False
)

print(
    "\nsaved -> results/chestxdet10_balsci_summary.csv"
)

