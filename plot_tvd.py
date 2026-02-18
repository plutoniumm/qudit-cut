import matplotlib.pyplot as plt
import pandas as pd

# PRE PROCESSING
cols = ["d1", "d2", "T", "terms", "tvd", "speedup"]
opt = pd.read_csv("./scripts/optimal.csv", engine="python")
opt.columns = cols
opt = opt.astype(float)
opt["trunc"] = 1e-6

opt["d1"] = opt["d1"].astype(int)
opt["d2"] = opt["d2"].astype(int)

# MAIN
df = opt.copy()

fig, ax = plt.subplots(figsize=(8, 5))

tvd = df["tvd"].clip(lower=0)
sizes = 100 * (tvd / tvd.max()) + 10

sc = ax.scatter(
    df["d1"],
    df["d2"],
    s=sizes,
    c=tvd,
    cmap="viridis",
    alpha=0.8,
    edgecolors="k",
    linewidths=0.6,
)

ax.set_xlabel("$d_1$", fontsize=14)
ax.set_ylabel("$d_2$", fontsize=14)

ax.set_xticks(sorted(df["d1"].unique()))
ax.set_yticks(sorted(df["d2"].unique()))

ax.tick_params(axis="both", which="both", labelsize=14)
ax.grid(True, which="both", linestyle="--", alpha=0.4)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("TVD", fontsize=12)

plt.tight_layout()
plt.savefig("./local/tvd.png", dpi=300)
