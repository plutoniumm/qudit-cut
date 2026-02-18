import matplotlib.pyplot as plt
import pandas as pd

# PRE PROCESSING
cols = ["d1", "d2", "T", "terms", "tvd", "speedup"]
opt = pd.read_csv("./scripts/truncimal.csv", engine="python")
opt.columns = cols
opt = opt.astype(float)

# MAIN
df = opt.copy()

agg = {}
count = {}
for _, row in df.iterrows():
    key = (row["d1"], row["d2"])
    agg[key] = agg.get(key, 0.0) + row["speedup"]
    count[key] = count.get(key, 0) + 1

avg_rows = []
for (d1, d2), total in agg.items():
    avg_rows.append({"d1": d1, "d2": d2, "speedup": total / count[(d1, d2)]})

df = pd.DataFrame(avg_rows)

df["dim"] = (df["d1"] * df["d2"])**2

df = df[df['d1'] <= 15]
df = df[df['d2'] <= 15]
df = df[df['dim'] >= 20]

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["dim"],
    df["speedup"],
    linewidth=1.2,
    alpha=0.8,
    marker="o",
)

ax.axhline(y=1, color='r', linestyle='--', linewidth=1.0, alpha=0.7)

ax.set_xlabel("CX dim=$d_1d_2$", fontsize=14)
ax.set_ylabel("speedup", fontsize=14)
ax.set_xscale("log")
ax.tick_params(axis="both", which="both", labelsize=14)
ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("./local/speedup.png", dpi=300)
