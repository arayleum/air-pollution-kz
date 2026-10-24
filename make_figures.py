"""Draws the figures of the Results section (paper Figs. 2-6) from the notebook outputs.

Usage:  python make_figures.py results results/paper_figures
        (first argument: folder with tables/ and figures/ written by the notebook)
All numbers come from the CSV files written by the notebook; the confusion-matrix
counts are the ones printed by the notebook (cell "RQ3: risk classes").
"""
import sys, os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
T = lambda f: pd.read_csv(os.path.join(RES, "tables", f))

MODELS = ["Baseline", "XGBoost", "Random Forest", "SVR", "LSTM"]
SHORT = {"Baseline": "Baseline", "XGBoost": "XGBoost", "Random Forest": "RF", "SVR": "SVR", "LSTM": "LSTM"}
# baseline = neutral reference gray; models = validated categorical slots 1-4
COL = {"Baseline": "#898781", "XGBoost": "#2a78d6", "Random Forest": "#eb6834",
       "SVR": "#1baf7a", "LSTM": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TASKS = [("pm25", 1), ("pm25", 24), ("pm10", 1), ("pm10", 24)]
TNAME = lambda p, h: f"{'PM2.5' if p == 'pm25' else 'PM10'}, {h} h ahead"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "axes.titleweight": "bold",
})

def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------- Fig. 2 RMSE + CI
m = T("table2_main_results.csv")
val = lambda s: float(s.split(" ± ")[0])
ci = lambda s: [float(x) for x in s.strip("[]").split(", ")]
fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.3))
for ax, (p, h) in zip(axes, TASKS):
    g = m[(m.pollutant == p) & (m.horizon == h)].set_index("model").loc[MODELS]
    y = g.RMSE.map(val).values
    lo, hi = np.array(g["RMSE 95% CI"].map(ci).tolist()).T
    x = np.arange(len(MODELS))
    ax.bar(x, y, width=0.72, color=[COL[k] for k in MODELS], edgecolor="white", linewidth=1)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axhline(y[0], color=COL["Baseline"], lw=0.8, ls="--")
    ax.set_xticks(x, [SHORT[k] for k in MODELS], rotation=45, ha="right")
    ax.set_title(TNAME(p, h)); style(ax); ax.set_ylim(0, hi.max() * 1.08)
axes[0].set_ylabel("Test RMSE (µg/m³)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_rmse_ci.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4 risk classification
CM = {"Baseline": np.array([[4988, 763, 177], [812, 811, 222], [151, 201, 647]]),
      "LSTM":     np.array([[4361, 1476, 91], [566, 1045, 234], [0, 415, 584]])}
r = T("table5_risk_classification.csv")
for k, cm in CM.items():  # consistency check against the CSV
    rec = cm.diagonal() / cm.sum(1)
    exp = r[r.model == k].set_index("class").loc[["safe", "moderate", "dangerous"], "recall"].values
    assert np.allclose(rec, exp, atol=1e-3), k
blues = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
fig = plt.figure(figsize=(7.16, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.55)
cls = ["safe", "moderate", "danger."]
for i, k in enumerate(["Baseline", "LSTM"]):
    ax = fig.add_subplot(gs[i]); cm = CM[k]; share = cm / cm.sum(1, keepdims=True)
    ax.imshow(share, cmap=blues, vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{cm[a, b]:,}\n{100 * share[a, b]:.0f}%", ha="center", va="center", fontsize=6.8,
                    color="white" if share[a, b] > 0.55 else INK)
    ax.set_xticks(range(3), cls); ax.set_yticks(range(3), cls)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Observed class")
    ax.set_title(f"({'ab'[i]}) {k}"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
ax = fig.add_subplot(gs[2])
d = r[r["class"] == "dangerous"].set_index("model").loc[MODELS]
for k in MODELS:
    ax.scatter(d.loc[k, "precision"], d.loc[k, "recall"], s=46, color=COL[k], edgecolor="white", lw=1.2, zorder=3)
    off = {"Baseline": (4, 3), "XGBoost": (5, -3), "Random Forest": (5, -6), "SVR": (5, -3), "LSTM": (5, 2)}[k]
    ax.annotate(SHORT[k], (d.loc[k, "precision"], d.loc[k, "recall"]), xytext=off, textcoords="offset points",
                fontsize=6.8, color=INK, va="center")
ax.set_xlim(0.60, 0.74); ax.set_ylim(0.45, 0.68)
ax.set_xlabel("Precision (dangerous class)"); ax.set_ylabel("Recall (dangerous class)")
ax.set_title("(c) Dangerous class"); style(ax, "both")
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.2, top=0.88)
fig.savefig(os.path.join(OUT, "fig4_risk_classification.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig. 5 by city
c = T("table6_by_city.csv"); cities = list(dict.fromkeys(c.city))
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
w = 0.8 / len(MODELS); x = np.arange(len(cities))
for ax, metric, title, ylab in [
        (axes[0], "MAE pm25", "(a) PM2.5 MAE, 24 h ahead", "Test MAE (µg/m³)"),
        (axes[1], "MAE pm10", "(b) PM10 MAE, 24 h ahead", "Test MAE (µg/m³)"),
        (axes[2], "recall dangerous", "(c) Dangerous-class Recall", "Recall")]:
    g = c[c.metric == metric].set_index("city").loc[cities]
    for i, k in enumerate(MODELS):
        ax.bar(x + (i - 2) * w, g[k].fillna(0).values, w, color=COL[k], edgecolor="white", lw=0.6, label=k)
    labels = cities
    ax.set_xticks(x, labels, rotation=30, ha="right", rotation_mode="anchor"); ax.set_title(title); ax.set_ylabel(ylab); style(ax)
    if metric == "recall dangerous":
        ax.set_ylim(0, 1)
        ax.text(x[cities.index("Pavlodar")], 0.04, "no dangerous\nhours", ha="center", fontsize=6, color=INK2)
        ax.text(x[cities.index("Temirtau")] + 0.15, 0.06, "0", ha="center", fontsize=6, color=INK2)
axes[0].legend(frameon=False, loc="upper left", handlelength=1, handleheight=0.8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_by_city.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3 Diebold-Mariano p-values
dm = T("table4_diebold_mariano.csv")
seq = LinearSegmentedColormap.from_list("s", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.25))
ab = {"Baseline": "B", "XGBoost": "X", "Random Forest": "R", "SVR": "S", "LSTM": "L"}
for ax, (p, h) in zip(axes, TASKS):
    g = dm[(dm.pollutant == p) & (dm.horizon == h)]
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5)
    for _, row in g.iterrows():
        i, j = MODELS.index(row["model B"]) - 1, MODELS.index(row["model A"])
        pv = row["p-value"]; sig = pv < 0.05
        strength = min(-np.log10(pv), 3) / 3
        face = seq(strength) if sig else "#f0efec"
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=1.5))
        txt = "<.001" if pv < 0.001 else f"{pv:.3f}"
        ax.text(j, i - 0.1, txt, ha="center", va="center", fontsize=6.3,
                color="white" if sig and strength > 0.5 else INK, fontweight="bold" if sig else "normal")
        ax.text(j, i + 0.22, ab[row["lower error"]], ha="center", va="center", fontsize=5.8,
                color="white" if sig and strength > 0.5 else INK2)
    ax.set_xticks(range(4), [SHORT[k] for k in MODELS[:4]], rotation=45, ha="right")
    ax.set_yticks(range(4), [SHORT[k] for k in MODELS[1:]])
    ax.tick_params(length=0); ax.set_title(TNAME(p, h))
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_dm_tests.pdf")); plt.close(fig)
# ---------------------------------------------------------------- Fig. 6 SHAP (drawn by the notebook)
shutil.copy(os.path.join(RES, "figures", "fig5_shap_xgb_pm25_h24.pdf"), os.path.join(OUT, "fig6_shap_xgb_pm25_h24.pdf"))
print("figures written to", OUT)
