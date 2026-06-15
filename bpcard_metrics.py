"""
BP-Card metrics: fill the standardized table and make the two recommended plots.

Usage:
    import pandas as pd
    df = pd.read_csv("your_data.csv")  # columns: ref_sbp, pred_sbp
    table = performance_table(df, "ref_sbp", "pred_sbp")
    print(table)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def compute_row(reference, prediction, baseline=None):
    y = np.asarray(reference, dtype=float)
    yhat = np.asarray(prediction, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[mask], yhat[mask]

    e = yhat - y
    md, sd = e.mean(), e.std(ddof=1)
    mad = np.abs(e).mean()
    mapd = 100 * (np.abs(e) / np.abs(y)).mean()
    cp5  = 100 * (np.abs(e) <=  5).mean()
    cp10 = 100 * (np.abs(e) <= 10).mean()
    cp15 = 100 * (np.abs(e) <= 15).mean()
    r, _ = pearsonr(y, yhat)

    row = {
        "N": len(y),
        "MD±SD (mmHg)": f"{md:.2f} ± {sd:.2f}",
        "MAD (mmHg)": round(mad, 2),
        "MAPD (%)": round(mapd, 2),
        "CP5 (%)": round(cp5, 1),
        "CP10 (%)": round(cp10, 1),
        "CP15 (%)": round(cp15, 1),
        "R": round(r, 3),
    }

    # Optional ΔBP column: BP variation relative to a baseline (e.g. calibration BP)
    if baseline is not None:
        base = np.asarray(baseline, dtype=float)[mask]
        delta = base - y   # ΔBP = baseline - current
        row = {"ΔBP±SD (mmHg)": f"{delta.mean():.2f} ± {delta.std(ddof=1):.2f}", **row}

    return row


def performance_table(df, reference, prediction, group=None, baseline=None, group_order=None):
    """Build the standardized performance table.

    group : optional column name to split rows by (e.g. BP category, or
            calibration condition: Static / Dynamic / Time after calibration)
    baseline : optional column name of baseline/calibration BP -> adds ΔBP column
    """
    rows, index = [], []
    base = df[baseline] if baseline else None

    rows.append(compute_row(df[reference], df[prediction], base))
    index.append("All")

    if group is not None:
        labels = group_order or sorted(df[group].dropna().unique())
        for g in labels:
            sub = df[df[group] == g]
            if len(sub) == 0:
                continue
            b = sub[baseline] if baseline else None
            rows.append(compute_row(sub[reference], sub[prediction], b))
            index.append(g)

    return pd.DataFrame(rows, index=index)


def plot_correlation(reference, prediction, ax=None, title=None):
    """Scatter of predicted vs reference, with identity line and fit."""
    y = np.asarray(reference, dtype=float)
    yhat = np.asarray(prediction, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(y, yhat, s=10, alpha=0.5, edgecolor="none")
    lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    m, b = np.polyfit(y, yhat, 1)
    ax.plot([lo, hi], [m*lo+b, m*hi+b], "r-", lw=1.5, label="fit")

    e = yhat - y
    r, _ = pearsonr(y, yhat)
    ax.set_title((title + "\n" if title else "")
                 + f"R = {r:.2f}   MD ± SD = {e.mean():.1f} ± {e.std(ddof=1):.1f} mmHg",
                 fontsize=10)
    ax.set_xlabel("Reference BP (mmHg)")
    ax.set_ylabel("Cuffless BP (mmHg)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    return ax


def plot_bland_altman(reference, prediction, ax=None, title=None):
    """Bland-Altman plot: difference vs mean, with bias and 95% limits of agreement."""
    y = np.asarray(reference, dtype=float)
    yhat = np.asarray(prediction, dtype=float)
    diff = yhat - y
    mean = (y + yhat) / 2

    md, sd = diff.mean(), diff.std(ddof=1)
    hi, lo = md + 1.96*sd, md - 1.96*sd

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    ax.scatter(mean, diff, s=10, alpha=0.5, edgecolor="none")
    ax.axhline(md, color="k", lw=1.2, label=f"bias = {md:.1f}")
    ax.axhline(hi, color="r", ls="--", lw=1, label=f"+1.96 SD = {hi:.1f}")
    ax.axhline(lo, color="r", ls="--", lw=1, label=f"-1.96 SD = {lo:.1f}")
    ax.set_title((title + "\n" if title else "")
                 + f"Mean diff ± SD = {md:.1f} ± {sd:.1f} mmHg", fontsize=10)
    ax.set_xlabel("Mean of Cuffless and Reference BP (mmHg)")
    ax.set_ylabel("Difference (Cuffless - Reference) (mmHg)")
    ax.legend(fontsize=8)
    return ax


if __name__ == "__main__":
    # --- demo with synthetic data ---
    rng = np.random.default_rng(0)
    n = 200
    ref = rng.normal(128, 18, n).clip(85, 200)
    pred = ref + rng.normal(2, 7, n)
    df = pd.DataFrame({"ref_sbp": ref, "pred_sbp": pred})

    table = performance_table(df, "ref_sbp", "pred_sbp")
    print(table.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    plot_correlation(df["ref_sbp"], df["pred_sbp"], ax=ax[0], title="Correlation")
    plot_bland_altman(df["ref_sbp"], df["pred_sbp"], ax=ax[1], title="Bland-Altman")
    fig.tight_layout()
    fig.savefig("example_plots.png", dpi=130)
    print("Saved example_plots.png")
