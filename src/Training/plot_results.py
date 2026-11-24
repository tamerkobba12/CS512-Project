"""plot_results.py - standard plots and metrics for the best model that MainMLR.py found.

What it does
  1. Reads the output file made by MainMLR.py and picks the best model (the lowest fitness among
     the models whose R2 of training, Q2 of training, R2 of validation and R2 of test all reach
     the bar, 0.6, or 0.5 if none does; the same rule MainMLR.py prints at the end).
  2. Fits that model (multiple linear regression) on the training molecules, predicts the
     training, validation and test molecules, and computes the metrics.
  3. Draws the figures, saves them as PNG files in the folder  plots/  and shows them.

Run it in the folder that holds the six data files (Train-Data.csv, Train-pIC50.csv, ...):
    python plot_results.py                      # newest output file of MainMLR.py in this folder
    python plot_results.py 2026-11-23-11-19-46.csv
    python plot_results.py --min-all 0.7        # a stricter bar: all four values >= 0.7
    python plot_results.py --no-show            # only save the PNG files, do not open windows

Figures
  1_metrics.png               R2 / Q2 of the four values the professor asked for + a table of metrics
  2_observed_vs_predicted.png observed against predicted pIC50: training, validation, test
  3_ga_progress.png           lowest fitness found so far while the genetic algorithm ran
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Look: one colour (and marker) per data set, the same in every figure.
# The marker shape repeats the colour, so the sets can be told apart without colour.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SETS = ("Training", "Validation", "Test")
COLOR = {"Training": "#2a78d6", "Validation": "#eb6834", "Test": "#1baf7a"}
MARKER = {"Training": "o", "Validation": "s", "Test": "^"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 9,
})


# ------------------------------------------------------------------------------
# 1. The data and the best model
def load_data(folder):
    """The six files MainMLR.py reads (no header row)."""
    DATA_FOLDER = Path(__file__).resolve().parents[2] / "data" / "processed"

    def read(name):
        return np.loadtxt(Path(folder) / name, delimiter=",", ndmin=2)
    X = {"Training": read(DATA_FOLDER / "Train-Data.csv"), "Validation": read(DATA_FOLDER / "Validation-Data.csv"),
         "Test": read(DATA_FOLDER / "Test-Data.csv")}
    y = {"Training": read(DATA_FOLDER / "Train-pIC50.csv").ravel(), "Validation": read(DATA_FOLDER / "Validation-pIC50.csv").ravel(),
         "Test": read(DATA_FOLDER / "Test-pIC50.csv").ravel()}
    return X, y


def pick_best_model(table, r2_train, q2_train, r2_valid, r2_test):
    """Lowest fitness among the models that meet the minimums; ties: fewest descriptors."""
    scores = table[["R2", "Q2", "R2Pred_Validation", "R2Pred_Test"]]
    ok = (scores >= [r2_train, q2_train, r2_valid, r2_test]).all(axis=1) & (scores < 1).all(axis=1)
    meets = bool(ok.any())
    pool = table[ok] if meets else table          # nothing qualifies: show the lowest fitness anyway
    pool = pool.drop_duplicates(subset="Descriptor ID")
    pool = pool.assign(NumDescriptors=pool["Descriptor ID"].str.count(",") + 1)
    best = pool[pool["Fitness"] == pool["Fitness"].min()].sort_values("NumDescriptors", kind="stable")
    return best.iloc[0], meets


# ------------------------------------------------------------------------------
# 2. The model and the metrics (all vectorized)
def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def fit_mlr(X, y):
    """Least squares with an intercept; returns the coefficients and (X'X)^-1."""
    Xd = add_intercept(X)
    coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
    return coef, np.linalg.pinv(Xd.T @ Xd)


def r2_own(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def r2_pred(y_train, y, yhat):
    """The professor's R2pred: the spread is measured around the TRAINING mean."""
    return 1 - ((y - yhat) ** 2).sum() / ((y - y_train.mean()) ** 2).sum()


def metrics_for(y_train, y, yhat, name):
    resid = y - yhat
    return {"Set": name, "n": len(y),
            "R2": r2_own(y, yhat) if name == "Training" else r2_pred(y_train, y, yhat),
            "RMSE": np.sqrt((resid ** 2).mean()), "MAE": np.abs(resid).mean(),
            "Pearson r": np.corrcoef(y, yhat)[0, 1]}


# ------------------------------------------------------------------------------
# 3. Helpers for the figures
def finish(fig, title, subtitle, name, outdir):
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.10 / height, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.44 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.4)
    fig.savefig(Path(outdir) / name, dpi=200)


def scatter_set(ax, x, y, name):
    ax.scatter(x, y, s=34, marker=MARKER[name], color=COLOR[name], edgecolor=SURFACE,
               linewidth=0.7, alpha=0.9, label=name, zorder=3)


def main():
    parser = argparse.ArgumentParser(description="Plots and metrics for the best model of MainMLR.py")
    parser.add_argument("output_file", nargs="?", help="the csv written by MainMLR.py (default: the newest one here)")
    parser.add_argument("--data-dir", default=".", help="folder with Train-Data.csv, ... (default: here)")
    parser.add_argument("--outdir", default="plots", help="folder for the PNG files (default: plots)")
    parser.add_argument("--min-all", type=float, default=None, metavar="X",
                        help="the bar: R2, Q2, R2 validation and R2 test all >= X "
                             "(default: 0.6, and 0.5 if no model reaches 0.6; the same as MainMLR.py)")
    parser.add_argument("--no-show", action="store_true", help="save the figures without opening windows")
    args = parser.parse_args()

    if args.output_file is None:                         # newest file named like 2026-11-23-11-19-46.csv
        candidates = sorted(p for p in Path(".").glob("*.csv") if re.fullmatch(r"\d{4}(-\d{2}){5}", p.stem))
        if not candidates:
            raise SystemExit("No output file of MainMLR.py found here; give its name: python plot_results.py <file>.csv")
        args.output_file = candidates[-1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- the best model -------------------------------------------------------------------
    table = pd.read_csv(args.output_file)
    for bar in ([args.min_all] if args.min_all is not None else [0.6, 0.5]):
        row, meets = pick_best_model(table, *[bar] * 4)
        rule = f"R2 train, Q2, validation and test all >= {bar}"
        if meets:
            break
    columns = [int(v) for v in re.findall(r"\d+", row["Descriptor ID"])]
    k = len(columns)
    first_found = int(table.index[table["Descriptor ID"] == row["Descriptor ID"]][0]) + 1
    print(f"Output file: {args.output_file}  ({len(table)} models)")
    if not meets:
        print(f"WARNING: no model meets the minimums ({rule}); showing the lowest fitness instead.")
    else:
        print(f"Rule used to pick the model: {rule}")
    print(f"Best model: fitness {row['Fitness']}, {k} descriptors (columns {columns})")

    X_all, y = load_data(args.data_dir)
    X = {name: X_all[name][:, columns] for name in SETS}

    # ---- fit on the training molecules, predict the three sets -------------------------------
    coef, XtX_inv = fit_mlr(X["Training"], y["Training"])
    yhat = {name: add_intercept(X[name]) @ coef for name in SETS}

    # Q2 (leave-one-out) of the training molecules without refitting 148 times:
    # for least squares the left-out residual is  residual / (1 - leverage)  (exact)
    Xd = add_intercept(X["Training"])
    leverage = np.einsum("ij,jk,ik->i", Xd, XtX_inv, Xd)                 # diagonal of the hat matrix
    loo_resid = (y["Training"] - yhat["Training"]) / (1 - leverage)
    q2 = 1 - (loo_resid ** 2).sum() / ((y["Training"] - y["Training"].mean()) ** 2).sum()

    metrics = pd.DataFrame([metrics_for(y["Training"], y[name], yhat[name], name) for name in SETS]).set_index("Set")
    print("\nMetrics (training: R2 = 1 - SSE/SST; validation and test: R2pred, spread around the training mean)")
    print(metrics.round(3).to_string())
    print(f"Q2 of training (leave-one-out): {q2:.3f}")
    print(f"In the output file (2 decimals): R2 {row['R2']}, Q2 {row['Q2']}, "
          f"R2 validation {row['R2Pred_Validation']}, R2 test {row['R2Pred_Test']}")
    metrics_out = metrics.copy()
    metrics_out["Q2 (leave-one-out)"] = [q2, np.nan, np.nan]
    metrics_out.round(4).to_csv(outdir / "model_metrics.csv")
    pd.concat([pd.DataFrame({"Set": name, "Molecule": np.arange(1, len(y[name]) + 1),
                             "Observed": y[name], "Predicted": yhat[name],
                             "Residual": y[name] - yhat[name]}) for name in SETS]
              ).round(4).to_csv(outdir / "predictions.csv", index=False)

    subtitle = (f"{k} descriptors, multiple linear regression | fitness {row['Fitness']} | "
                f"{len(y['Training'])} training / {len(y['Validation'])} validation / {len(y['Test'])} test molecules")

    # ---- Figure 1: the professor's four numbers and the metrics table -----------------------------
    fig = plt.figure(figsize=(13, 5.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.06, right=0.985, top=0.80, bottom=0.13, wspace=0.12)
    ax = fig.add_subplot(grid[0])
    bars = [("R² training", metrics.loc["Training", "R2"], "Training", False),
            ("Q² training\n(leave-one-out)", q2, "Training", True),
            ("R² validation", metrics.loc["Validation", "R2"], "Validation", False),
            ("R² test", metrics.loc["Test", "R2"], "Test", False)]
    for i, (label, value, name, hatched) in enumerate(bars):
        ax.bar(i, value, width=0.55, color=COLOR[name], alpha=0.55 if hatched else 1, hatch="///" if hatched else None,
               edgecolor=COLOR[name], linewidth=0.8, zorder=3)
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks(range(4), [b[0] for b in bars])
    ax.set_xlim(-0.5, 4.35)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("R² / Q²")
    ax.grid(axis="x", visible=False)
    for level, text in ((0.6, "0.6  (her code)"), (0.5, "0.5  (slides)")):
        ax.hlines(level, -0.5, 3.38, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        ax.text(3.45, level, text, ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set_title("The four values the professor asked for")

    ax = fig.add_subplot(grid[1])
    ax.axis("off")
    cells = [["R²  (R²pred for validation, test)"] + [f"{metrics.loc[s, 'R2']:.3f}" for s in SETS],
             ["Q²  (leave-one-out)", f"{q2:.3f}", "–", "–"],
             ["RMSE"] + [f"{metrics.loc[s, 'RMSE']:.3f}" for s in SETS],
             ["MAE"] + [f"{metrics.loc[s, 'MAE']:.3f}" for s in SETS],
             ["Pearson r"] + [f"{metrics.loc[s, 'Pearson r']:.3f}" for s in SETS],
             ["Molecules"] + [str(int(metrics.loc[s, 'n'])) for s in SETS]]
    tab = ax.table(cellText=cells, colLabels=["Metric", "Training", "Validation", "Test"], loc="center",
                   cellLoc="center", colWidths=[0.46, 0.18, 0.18, 0.18])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.9)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK)
        if c == 0:
            cell.get_text().set_ha("left")
            cell.get_text().set_color(INK2)
        if r == 0:
            cell.get_text().set_fontweight("bold")
            cell.set_linewidth(0)
    ax.set_title("Metrics", loc="left")
    finish(fig, "Best model: R², Q², RMSE, MAE and r", subtitle, "1_metrics.png", outdir)

    # ---- Figure 2: observed against predicted ------------------------------------------------------
    lo = min(min(y[s].min(), yhat[s].min()) for s in SETS) - 0.3      # shared limits for the three panels
    hi = max(max(y[s].max(), yhat[s].max()) for s in SETS) + 0.3
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.12, wspace=0.08)
    for ax, name in zip(axes, SETS):
        ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1.0, zorder=2)                         # perfect prediction
        scatter_set(ax, y[name], yhat[name], name)
        m = metrics.loc[name]
        r2_label = "R²" if name == "Training" else "R²pred"
        text = f"{r2_label} = {m['R2']:.2f}\nRMSE = {m['RMSE']:.2f}\nMAE = {m['MAE']:.2f}"
        if name == "Training":
            text += f"\nQ² = {q2:.2f}"
        ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=INK, linespacing=1.5)
        ax.set_title(f"{name}  (n = {int(m['n'])})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed pIC50")
    axes[0].set_ylabel("Predicted pIC50")
    axes[2].legend(handles=[plt.Line2D([], [], color=INK2, linewidth=1.0, label="perfect prediction")], loc="lower right")
    finish(fig, "Observed against predicted pIC50", subtitle, "2_observed_vs_predicted.png", outdir)

    # ---- Figure 3: how the search went ------------------------------------------------------------
    running_best = table["Fitness"].cummin().to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.80, bottom=0.14)
    ax.plot(np.arange(1, len(table) + 1), running_best, color=COLOR["Training"], linewidth=2, drawstyle="steps-post")
    ax.scatter([first_found], [row["Fitness"]], s=46, color=COLOR["Training"], edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.annotate(f"final model first scored\nas model {first_found:,} (fitness {row['Fitness']})",
                (first_found, row["Fitness"]), xytext=(-12, 24), textcoords="offset points", ha="right", fontsize=9,
                color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_xlim(0, len(table))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Models scored (order of the output file)")
    ax.set_ylabel("Lowest fitness so far")
    ax.set_title("Genetic algorithm progress (lower is better)")
    finish(fig, "Fitness of the best model so far", f"{len(table):,} models scored in total", "3_ga_progress.png", outdir)

    print(f"\nFigures, model_metrics.csv and predictions.csv are in the folder: {outdir.resolve()}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()