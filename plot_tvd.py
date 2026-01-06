import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./grid.log", names=["D", "T", "TVD", "Slow", "Terms"], header=None, engine="python")
df = df[df["D"].astype(str).str.match(r"\(\d+\s+\d+\)")].copy()

d_vals = df["D"].str.extract(r"\((\d+)\s+(\d+)\)").astype(int)
df["dim"] = 2 ** d_vals.sum(axis=1)

df["Slow"] = pd.to_numeric(df["Slow"].astype(str).str.rstrip("x"), errors="coerce")
df["T"] = pd.to_numeric(df["T"], errors="coerce")
df["TVD"] = pd.to_numeric(df["TVD"], errors="coerce")
df = df.dropna()

fig, ax = plt.subplots(figsize=(8, 5))

for t, group in df.groupby("T"):
    group = group.sort_values("dim")
    ax.scatter(group["dim"], group["Slow"], s=group["TVD"] * 2000 + 20, alpha=0.6)
    ax.plot(group["dim"], group["Slow"], linewidth=1.2, alpha=0.6, label=f"{t:.0e}")

ax.set_xlabel("dim=$(d_1d_2)^2$", fontsize=14)
ax.set_ylabel("time (s)", fontsize=14)
ax.set_xscale("log")
ax.tick_params(axis="both", which="both", labelsize=14)
ax.grid(True, which="both", linestyle="--", alpha=0.4)

legend1 = ax.legend(title="Truncation threshold", fontsize=14, title_fontsize=14, loc="upper left")
ax.add_artist(legend1)

size_tvd_vals = [0.0, 1e-3, 1e-2, 1e-1]
size_handles = [
    plt.scatter([], [], s=(val * 2000 + 20 if val > 0 else 20), color="gray", alpha=0.6)
    for val in size_tvd_vals
]
size_labels = ["TVD = 0", "TVD = 1e-3", "TVD = 1e-2", "TVD = 1e-1"]
ax.legend(
    size_handles,
    size_labels,
    title="Bubble size",
    fontsize=14,
    title_fontsize=14,
    loc="upper center",
    scatterpoints=1,
    frameon=True,
)

plt.tight_layout()
plt.savefig("./local/scaling.png", dpi=300)