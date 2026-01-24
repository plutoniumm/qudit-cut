import matplotlib.pyplot as plt
import pandas as pd

cols = ["d1", "d2", "T", "terms", "TVD", "Slow"]

opt = pd.read_csv("./scripts/optimal.csv", engine="python")
opt.columns = cols
opt = opt.astype(float)
opt["trunc"] = 1e-6

tri = pd.read_csv("./scripts/truncimal.csv", engine="python")
tri.columns = cols
tri = tri.astype(float)
tri["trunc"] = 2e-4

df = pd.concat([opt, tri], ignore_index=True)
df["dim"] = (df["d1"] * df["d2"])

fig, ax = plt.subplots(figsize=(8, 5))
for trunc, group in df.groupby("trunc"):
    group = group.sort_values("dim")
    ax.plot(group["dim"], group["TVD"], marker="o", linewidth=1.5, alpha=0.5, label=f"{trunc:.0e}")

ax.set_xlabel("dim=$d_1d_2$", fontsize=14)
ax.set_ylabel("TVD", fontsize=14)
ax.set_xscale("log")
ax.tick_params(axis="both", which="both", labelsize=14)
ax.grid(True, which="both", linestyle="--", alpha=0.4)

ax.legend(title="Truncation", fontsize=14, title_fontsize=14, loc="best")

plt.tight_layout()
# plt.savefig("./local/scaling.png", dpi=300)
plt.show()
