import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

experiments_dir = Path(__file__).parent.parent
results_dir = experiments_dir / "results"
output_dir = results_dir / "analysis/eval1"
csv_filename = output_dir / "parsed_scores.csv"
img_filename = output_dir / "score_distributions.png"

file_name = results_dir / "exp1.json"

with open(file_name, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for k, v in data.items():
    eavl = v.get("eavl", "")

    match = re.search(r"檢索\s*(\d+)\s*分，生成\s*(\d+)\s*分", eavl)
    if match:
        retrieval = int(match.group(1))
        generation = int(match.group(2))
    else:
        nums = re.findall(r"\d+", eavl)
        if len(nums) >= 2:
            retrieval = int(nums[0])
            generation = int(nums[1])
        else:
            retrieval = None
            generation = None

    records.append(
        {"id": int(k), "retrieval_score": retrieval, "generation_score": generation}
    )

df = pd.DataFrame(records)

df.to_csv(csv_filename, index=False)
print(f"已儲存解析後的資料至 {csv_filename}")

print("\n--- 敘述性統計資料 ---")
print(df.describe())

print("\n--- 檢索分數分佈 ---")
print(df["retrieval_score"].value_counts().sort_index())

print("\n--- 生成分數分佈 ---")
print(df["generation_score"].value_counts().sort_index())

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(
    x="retrieval_score",
    data=df,
    ax=axes[0],
    color="skyblue",
    order=sorted(df["retrieval_score"].dropna().unique()),
)
axes[0].set_title("Retrieval Score Distribution")
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Count")

sns.countplot(
    x="generation_score",
    data=df,
    ax=axes[1],
    color="salmon",
    order=sorted(df["generation_score"].dropna().unique()),
)
axes[1].set_title("Generation Score Distribution")
axes[1].set_xlabel("Score")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(img_filename, dpi=300)
print(f"圖表已儲存為 {img_filename}")
