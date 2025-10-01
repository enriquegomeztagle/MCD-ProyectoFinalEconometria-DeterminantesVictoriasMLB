# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

# %%
MASTER = Path("../data/processed/master_teams_2000_2019.csv")
TAB_OUT = Path("../data/processed")
FIG_OUT = Path("../plots")
TAB_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

# %%
df = pd.read_csv(MASTER, parse_dates=["season_date"])
print("df head:")
df.head()

# %%
target = "W"
predictors = ["RunDiff", "ERA", "HR", "logHR1"]
cols_for_matrix = [target] + predictors

# %%
pearson_mat = df[cols_for_matrix].corr(method="pearson")
pearson_mat.to_csv(TAB_OUT / "corr_matrix_pearson.csv", float_format="%.6f")
print("pearson_mat:")
print(pearson_mat)

# %%
spearman_mat = df[cols_for_matrix].corr(method="spearman")
spearman_mat.to_csv(TAB_OUT / "corr_matrix_spearman.csv", float_format="%.6f")
print("spearman_mat:")
print(spearman_mat)

# %%
rows = []
for x in predictors:
    sub = df[[target, x]].dropna()
    r_p, p_p = pearsonr(sub[target], sub[x])
    r_s, p_s = spearmanr(sub[target], sub[x])
    rows.append(
        {
            "X": x,
            "pearson_r": r_p,
            "pearson_p": p_p,
            "spearman_rho": r_s,
            "spearman_p": p_s,
            "N": len(sub),
        }
    )

corr_wp = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
corr_wp.to_csv(TAB_OUT / "corr_W_vs_X.csv", index=False, float_format="%.6f")
print("corr w vs x:")
print(corr_wp)


# %%
def plot_heatmap(mat: pd.DataFrame, title: str, outpath: Path):
    data = mat.values
    labels = mat.columns.tolist()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coeficiente de correlación", rotation=90)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, transparent=True)
    plt.show()
    plt.close(fig)


plot_heatmap(
    pearson_mat, "Matriz de correlación (Pearson)", FIG_OUT / "heatmap_corr_pearson.png"
)
plot_heatmap(
    spearman_mat,
    "Matriz de correlación (Spearman)",
    FIG_OUT / "heatmap_corr_spearman.png",
)
print("Heatmaps guardados en plots/")

# %%
abs_r = corr_wp[["X", "pearson_r"]].copy()
abs_r["abs_pearson_r"] = abs_r["pearson_r"].abs()
abs_r = abs_r.sort_values("abs_pearson_r", ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(abs_r["X"], abs_r["abs_pearson_r"])
ax.set_xlabel("|r| con W (Pearson)")
ax.set_title("Fuerza de asociación absoluta con W")
for i, v in enumerate(abs_r["abs_pearson_r"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center")
fig.tight_layout()
fig.savefig(FIG_OUT / "bar_abs_r_W.png", dpi=150, transparent=True)
plt.show()
plt.close(fig)

# %%
try:
    with open(TAB_OUT / "corr_W_vs_X.tex", "w") as f:
        f.write(
            corr_wp.rename(
                columns={
                    "X": "Variable",
                    "pearson_r": "Pearson r",
                    "pearson_p": "p (Pearson)",
                    "spearman_rho": "Spearman $\\rho$",
                    "spearman_p": "p (Spearman)",
                    "N": "N",
                }
            ).to_latex(
                index=False,
                float_format="%.4f",
                caption="Correlación de W con variables explicativas",
                label="tab:corr_w_x",
            )
        )
    print("Tabla LaTeX exportada: data/processed/corr_W_vs_X.tex")
except Exception as e:
    print("No se exportó LaTeX (opcional). Razón:", e)

# %%
