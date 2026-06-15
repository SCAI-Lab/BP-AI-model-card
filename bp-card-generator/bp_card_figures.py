"""
Reusable BP-Card style figure builders.

The module is organized around four steps:
1. Load BP data from CSV into a BPCardData object.
2. Compute table metrics from raw CSV rows.
3. Draw individual BP-Card panels on Matplotlib axes.
4. Save each panel as a separate figure file.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import matplotlib.pyplot as plt


ArrayLike = Union[Iterable[float], np.ndarray]
DEFAULT_GROUP_ORDER = [
    "All",
    "Normal",
    "Prehypertension",
    "Hypertension Stage 1",
    "Hypertension Stage 2",
]
DEFAULT_CONDITION_ORDER = [
    "Static",
    "Dynamic",
    "Time after calibration",
]
TABLE_METRIC_COLUMNS = [
    "DeltaBP+/-SD (mmHg)",
    "MD+/-SD (mmHg)",
    "MAD (mmHg)",
    "MAPD (%)",
    "CP5 (%)",
    "CP10 (%)",
    "CP15 (%)",
    "R",
]


@dataclass(frozen=True)
class BPCardData:
    """Numeric arrays used to build BP-Card panels.

    Attributes:
        ref_train: Reference BP values for the training set.
        ref_test: Reference BP values for the test set.
        calib_train: Calibration or baseline BP values for the training set.
        calib_test: Calibration or baseline BP values for the test set.
        pred_train: Model/cuffless BP predictions for the training set.
        pred_test: Model/cuffless BP predictions for the test set.
    """

    ref_train: np.ndarray
    ref_test: np.ndarray
    calib_train: np.ndarray
    calib_test: np.ndarray
    pred_train: np.ndarray
    pred_test: np.ndarray


@dataclass(frozen=True)
class FigureStyle:
    """Visual styling shared by BP-Card figures.

    Attributes:
        prediction_color: Color used for prediction/correlation panels.
        calibration_color: Color used for calibration/change panels.
        train_hist_color: Histogram color for training absolute BP.
        change_hist_color: Histogram color for BP-change values.
        test_hist_color: Histogram color for test data.
        marker_size: Scatter marker size.
        marker_alpha: Scatter marker transparency.
        edge_color: Scatter marker edge color.
        edge_width: Scatter marker edge line width.
        grid_alpha: Transparency of dotted grid lines.
    """

    prediction_color: str = "#a31f34"
    calibration_color: str = "#1f4e79"
    train_hist_color: str = "#e07a8b"
    change_hist_color: str = "#9dc3e6"
    test_hist_color: str = "#9a9a9a"
    marker_size: float = 14
    marker_alpha: float = 0.8
    edge_color: str = "black"
    edge_width: float = 0.3
    grid_alpha: float = 0.4


def as_clean_arrays(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert two equal-shaped 1-D inputs to float arrays and drop paired NaNs.

    Args:
        x: First numeric array-like input.
        y: Second numeric array-like input.

    Returns:
        Two NumPy arrays with entries removed wherever either input had NaN.

    Raises:
        ValueError: If the two inputs do not have the same shape.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch: {x_arr.shape} vs {y_arr.shape}")

    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    return x_arr[mask], y_arr[mask]


def make_synthetic_bp_data(
    seed: int = 1,
    n_train: int = 600,
    n_test: int = 220,
    train_mean: float = 120,
    test_mean: float = 122,
    bp_sd: float = 15,
    bp_min: float = 85,
    bp_max: float = 180,
    calibration_sd: float = 5,
    prediction_train_bias: float = 0.2,
    prediction_test_bias: float = 0.5,
    prediction_train_sd: float = 5,
    prediction_test_sd: float = 5.9,
) -> BPCardData:
    """Create demo data for all six BP-Card style panels.

    This is only for examples. Real analyses should usually use
    `load_bp_card_data_from_csv`.

    Args:
        seed: Random seed for reproducibility.
        n_train: Number of training samples.
        n_test: Number of test samples.
        train_mean: Mean reference BP for training samples.
        test_mean: Mean reference BP for test samples.
        bp_sd: Standard deviation of reference BP values.
        bp_min: Lower clipping limit for reference BP.
        bp_max: Upper clipping limit for reference BP.
        calibration_sd: Standard deviation of calibration noise.
        prediction_train_bias: Mean prediction error in the training set.
        prediction_test_bias: Mean prediction error in the test set.
        prediction_train_sd: Standard deviation of training prediction error.
        prediction_test_sd: Standard deviation of test prediction error.

    Returns:
        BPCardData object containing reference, calibration, and prediction arrays.
    """
    rng = np.random.default_rng(seed)

    ref_train = rng.normal(train_mean, bp_sd, n_train).clip(bp_min, bp_max)
    ref_test = rng.normal(test_mean, bp_sd, n_test).clip(bp_min, bp_max)

    calib_train = ref_train + rng.normal(0, calibration_sd, n_train)
    calib_test = ref_test + rng.normal(0.5, calibration_sd, n_test)

    pred_train = ref_train + rng.normal(
        prediction_train_bias, prediction_train_sd, n_train
    )
    pred_test = ref_test + rng.normal(prediction_test_bias, prediction_test_sd, n_test)

    return BPCardData(
        ref_train=ref_train,
        ref_test=ref_test,
        calib_train=calib_train,
        calib_test=calib_test,
        pred_train=pred_train,
        pred_test=pred_test,
    )


def load_bp_card_data_from_csv(
    csv_path: str | Path,
    mode: str = "calibration_based",
    split_col: str = "split",
    reference_col: str = "reference_bp",
    prediction_col: str = "prediction_bp",
    calibration_col: str = "calibration_bp",
) -> BPCardData:
    """Load train/test BP arrays from a CSV file.

    Expected columns:
    - split: train or test
    - reference_bp: reference blood pressure
    - prediction_bp: cuffless/model blood pressure
    - calibration_bp: required for mode="calibration_based"

    For mode="calibration_free", calibration_bp may be omitted. The loader then
    uses the train reference mean as the baseline needed by panels (b), (e), (f).

    Args:
        csv_path: Path to the input CSV file.
        mode: Either "calibration_based" or "calibration_free".
        split_col: Column containing "train" and "test" labels.
        reference_col: Column containing reference BP values.
        prediction_col: Column containing model/cuffless BP values.
        calibration_col: Column containing calibration BP values. Required for
            calibration-based mode.

    Returns:
        BPCardData object ready for figure-generation functions.

    Raises:
        ValueError: If required columns or train/test rows are missing.
    """
    csv_path = Path(csv_path)
    if mode not in {"calibration_based", "calibration_free"}:
        raise ValueError("mode must be 'calibration_based' or 'calibration_free'")

    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"{csv_path} has no data rows")

    required = {split_col, reference_col, prediction_col}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    if mode == "calibration_based" and calibration_col not in rows[0]:
        raise ValueError(f"{csv_path} is missing required column: {calibration_col}")

    train_rows = [row for row in rows if row[split_col].strip().lower() == "train"]
    test_rows = [row for row in rows if row[split_col].strip().lower() == "test"]
    if not train_rows or not test_rows:
        raise ValueError(f"{csv_path} must contain both train and test rows")

    def values(selected_rows: list[dict[str, str]], column: str) -> np.ndarray:
        return np.asarray([float(row[column]) for row in selected_rows], dtype=float)

    ref_train = values(train_rows, reference_col)
    ref_test = values(test_rows, reference_col)
    pred_train = values(train_rows, prediction_col)
    pred_test = values(test_rows, prediction_col)

    if mode == "calibration_based":
        calib_train = values(train_rows, calibration_col)
        calib_test = values(test_rows, calibration_col)
    else:
        baseline = float(np.nanmean(ref_train))
        calib_train = np.full_like(ref_train, baseline)
        calib_test = np.full_like(ref_test, baseline)

    return BPCardData(
        ref_train=ref_train,
        ref_test=ref_test,
        calib_train=calib_train,
        calib_test=calib_test,
        pred_train=pred_train,
        pred_test=pred_test,
    )


def mean_sd(values: ArrayLike, ddof: int = 1) -> tuple[float, float]:
    """Return mean and standard deviation, ignoring NaNs.

    Args:
        values: Numeric values.
        ddof: Delta degrees of freedom passed to `np.nanstd`.

    Returns:
        Tuple `(mean, sd)`.
    """
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=ddof))


def correlation_summary(reference: ArrayLike, prediction: ArrayLike) -> dict[str, float]:
    """Compute correlation-panel summary statistics.

    Args:
        reference: Reference BP values.
        prediction: Model/cuffless BP predictions.

    Returns:
        Dictionary with Pearson `r`, mean signed difference, and SD of signed
        difference. Difference is `prediction - reference`.
    """
    ref, pred = as_clean_arrays(reference, prediction)
    diff = pred - ref
    r = np.nan if ref.size < 2 else float(np.corrcoef(ref, pred)[0, 1])
    md, sd = mean_sd(diff)
    return {"r": r, "mean_difference": md, "sd_difference": sd}


def bland_altman_summary(difference: ArrayLike) -> dict[str, float]:
    """Compute Bland-Altman summary statistics.

    Args:
        difference: Signed difference values for the y-axis.

    Returns:
        Dictionary with mean difference, SD, upper LoA, and lower LoA.
    """
    diff = np.asarray(difference, dtype=float)
    md, sd = mean_sd(diff)
    return {
        "mean_difference": md,
        "sd_difference": sd,
        "upper_loa": md + 1.96 * sd,
        "lower_loa": md - 1.96 * sd,
    }


def metric_row(
    reference: ArrayLike,
    prediction: ArrayLike,
    baseline: ArrayLike,
) -> dict[str, str | int | float]:
    """Compute one row of the standardized BP-Card performance table.

    Args:
        reference: Reference BP values.
        prediction: Model/cuffless BP predictions.
        baseline: Calibration or baseline BP values used to compute DeltaBP.

    Returns:
        Dictionary containing DeltaBP+/-SD, MD+/-SD, MAD, MAPD, CP5, CP10,
        CP15, and R.

    Raises:
        ValueError: If baseline shape differs from reference/prediction shape.
    """
    ref, pred = as_clean_arrays(reference, prediction)
    base = np.asarray(baseline, dtype=float)
    if base.shape != ref.shape:
        raise ValueError(f"baseline shape mismatch: {base.shape} vs {ref.shape}")

    error = pred - ref
    abs_error = np.abs(error)
    delta = ref - base
    delta_mean, delta_sd = mean_sd(delta)
    md, sd = mean_sd(error)
    denominator = np.where(ref == 0, np.nan, np.abs(ref))
    mapd = float(np.nanmean(abs_error / denominator) * 100.0)
    r = np.nan if ref.size < 2 else float(np.corrcoef(ref, pred)[0, 1])

    return {
        "DeltaBP+/-SD (mmHg)": f"{delta_mean:.2f} +/- {delta_sd:.2f}",
        "MD+/-SD (mmHg)": f"{md:.2f} +/- {sd:.2f}",
        "MAD (mmHg)": round(float(np.nanmean(abs_error)), 2),
        "MAPD (%)": round(mapd, 2),
        "CP5 (%)": round(float(np.nanmean(abs_error <= 5) * 100.0), 1),
        "CP10 (%)": round(float(np.nanmean(abs_error <= 10) * 100.0), 1),
        "CP15 (%)": round(float(np.nanmean(abs_error <= 15) * 100.0), 1),
        "R": round(r, 3),
    }


def performance_reporting_table_from_csv(
    csv_path: str | Path,
    mode: str = "calibration_free",
    group_col: str = "group",
    split_col: str = "split",
    split_value: str | None = None,
    reference_col: str = "reference_bp",
    prediction_col: str = "prediction_bp",
    calibration_col: str = "calibration_bp",
    baseline_col: str | None = None,
    baseline_value: float | None = None,
    group_order: list[str] | None = None,
    include_empty_groups: bool = True,
) -> list[dict[str, str | int | float]]:
    """Build the standardized table from CSV rows with pre-existing groups.

    The function does not calculate BP categories. It only uses values already
    present in `group_col`.

    Args:
        csv_path: Path to the input CSV file.
        mode: Either "calibration_free" or "calibration_based".
        group_col: CSV column containing group/condition labels to report.
        split_col: CSV column containing train/test labels.
        split_value: Optional split filter, such as "test". If None, all rows
            are used.
        reference_col: CSV column containing reference BP values.
        prediction_col: CSV column containing model/cuffless BP values.
        calibration_col: CSV column containing calibration BP values. Required
            for calibration-based mode.
        baseline_col: Optional CSV column containing baseline values for
            calibration-free mode.
        baseline_value: Optional scalar baseline for calibration-free mode when
            `baseline_col` is not provided. If None, each group uses its own
            reference mean as the baseline.
        group_order: Optional explicit output row order. For calibration-free
            tables use `DEFAULT_GROUP_ORDER`; for calibration-based tables use
            `DEFAULT_CONDITION_ORDER`.
        include_empty_groups: If True, include ordered groups even when no CSV
            rows exist for that label.

    Returns:
        List of dictionaries, one per table row, ready for pandas or CSV export.

    Raises:
        ValueError: If required columns are missing or filtering removes all rows.
    """
    csv_path = Path(csv_path)
    if mode not in {"calibration_based", "calibration_free"}:
        raise ValueError("mode must be 'calibration_based' or 'calibration_free'")

    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")

    required = {reference_col, prediction_col, group_col}
    if split_value is not None:
        required.add(split_col)
    if mode == "calibration_based":
        required.add(calibration_col)
    if baseline_col is not None:
        required.add(baseline_col)

    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    if split_value is not None:
        rows = [
            row for row in rows
            if row[split_col].strip().lower() == split_value.strip().lower()
        ]
        if not rows:
            raise ValueError(f"{csv_path} has no rows with {split_col}={split_value!r}")

    def values(selected_rows: list[dict[str, str]], column: str) -> np.ndarray:
        return np.asarray([float(row[column]) for row in selected_rows], dtype=float)

    def baseline_values(selected_rows: list[dict[str, str]]) -> np.ndarray:
        ref = values(selected_rows, reference_col)
        if mode == "calibration_based":
            return values(selected_rows, calibration_col)
        if baseline_col is not None:
            return values(selected_rows, baseline_col)
        baseline = float(np.nanmean(ref)) if baseline_value is None else baseline_value
        return np.full_like(ref, baseline)

    order = group_order or DEFAULT_GROUP_ORDER
    output_rows: list[dict[str, str | int | float]] = []

    row_groups = [row[group_col] for row in rows]
    groups = [group for group in order if group == "All" or group in row_groups]
    if not include_empty_groups:
        remaining = sorted(set(row_groups) - set(groups))
        groups.extend(remaining)

    for group in groups:
        selected = rows if group == "All" else [row for row in rows if row[group_col] == group]
        if not selected:
            if include_empty_groups:
                empty = {"Group": group}
                empty.update({key: "" for key in TABLE_METRIC_COLUMNS})
                output_rows.append(empty)
            continue

        table_row = {"Group": group}
        table_row.update(
            metric_row(
                reference=values(selected, reference_col),
                prediction=values(selected, prediction_col),
                baseline=baseline_values(selected),
            )
        )
        output_rows.append(table_row)

    return output_rows


def save_performance_table_csv(
    table_rows: list[dict[str, str | int | float]],
    output_path: str | Path,
) -> Path:
    """Save a standardized performance table as a CSV file.

    Args:
        table_rows: Rows returned by `performance_reporting_table_from_csv`.
        output_path: Destination CSV path.

    Returns:
        Path to the saved CSV file.

    Raises:
        ValueError: If `table_rows` is empty.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not table_rows:
        raise ValueError("table_rows is empty")

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    return output_path


def add_annotation_box(
    ax: plt.Axes,
    lines: Iterable[str],
    x: float = 0.02,
    y: float = 0.98,
    fontsize: float = 8,
) -> None:
    """Add a white annotation box in axis-relative coordinates.

    Args:
        ax: Matplotlib axis to annotate.
        lines: Text lines to place in the annotation box.
        x: Axis-relative x-position, where 0 is left and 1 is right.
        y: Axis-relative y-position, where 0 is bottom and 1 is top.
        fontsize: Annotation text size.
    """
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "square,pad=0.4",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    )


def finish_panel(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    style: FigureStyle,
) -> plt.Axes:
    """Apply labels, title, and grid shared by most panels.

    Args:
        ax: Matplotlib axis to format.
        title: Panel title, usually "(a)" through "(f)".
        xlabel: x-axis label.
        ylabel: y-axis label.
        style: Shared figure styling options.

    Returns:
        The formatted Matplotlib axis.
    """
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=style.grid_alpha)
    return ax


def plot_correlation_panel(
    ax: plt.Axes,
    reference: ArrayLike,
    prediction: ArrayLike,
    title: str,
    xlabel: str = "Reference BP (mmHg)",
    ylabel: str = "Cuffless BP (mmHg)",
    style: FigureStyle | None = None,
    point_color: str | None = None,
    show_identity: bool = True,
    show_fit: bool = False,
    open_markers: bool = False,
) -> plt.Axes:
    """Draw a BP-Card correlation panel, such as (c) or (e).

    Args:
        ax: Matplotlib axis to draw on.
        reference: x-axis reference or delta-reference values.
        prediction: y-axis prediction or delta-prediction values.
        title: Panel title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        style: Optional shared style. Defaults to `FigureStyle()`.
        point_color: Optional override for scatter marker color.
        show_identity: If True, draw the y=x identity line.
        show_fit: If True, draw a least-squares fit line.
        open_markers: If True, draw markers with no fill color.

    Returns:
        The Matplotlib axis containing the panel.
    """
    style = style or FigureStyle()
    ref, pred = as_clean_arrays(reference, prediction)
    color = point_color or style.prediction_color

    facecolor = "none" if open_markers else color
    ax.scatter(
        ref,
        pred,
        s=style.marker_size,
        facecolor=facecolor,
        edgecolor=color if open_markers else style.edge_color,
        linewidth=0.6 if open_markers else style.edge_width,
        alpha=style.marker_alpha,
    )

    lo = float(min(ref.min(), pred.min()))
    hi = float(max(ref.max(), pred.max()))
    if show_identity:
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    if show_fit:
        slope, intercept = np.polyfit(ref, pred, 1)
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                color="black", lw=1)

    stats = correlation_summary(ref, pred)
    add_annotation_box(
        ax,
        [
            f"Correlation (r) = {stats['r']:.2f}",
            "Mean Difference +/- SD = "
            f"{stats['mean_difference']:.1f} +/- {stats['sd_difference']:.1f} mmHg",
        ],
    )
    return finish_panel(ax, title, xlabel, ylabel, style)


def plot_bland_altman_panel(
    ax: plt.Axes,
    x_values: ArrayLike,
    difference: ArrayLike,
    title: str,
    xlabel: str,
    ylabel: str,
    style: FigureStyle | None = None,
    point_color: str | None = None,
    loa_band_half_width: float = 0.6,
) -> plt.Axes:
    """Draw a Bland-Altman style panel with bias and agreement limits.

    Args:
        ax: Matplotlib axis to draw on.
        x_values: x-axis values, typically a mean BP value per sample.
        difference: y-axis signed differences.
        title: Panel title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        style: Optional shared style. Defaults to `FigureStyle()`.
        point_color: Optional override for scatter marker color.
        loa_band_half_width: Half-height of the shaded LoA bands in mmHg.

    Returns:
        The Matplotlib axis containing the panel.
    """
    style = style or FigureStyle()
    x = np.asarray(x_values, dtype=float)
    diff = np.asarray(difference, dtype=float)
    x, diff = as_clean_arrays(x, diff)

    color = point_color or style.calibration_color
    ax.scatter(
        x,
        diff,
        s=style.marker_size,
        facecolor=color,
        edgecolor=style.edge_color,
        linewidth=style.edge_width,
        alpha=0.7,
    )

    stats = bland_altman_summary(diff)
    md = stats["mean_difference"]
    sd = stats["sd_difference"]
    upper = stats["upper_loa"]
    lower = stats["lower_loa"]

    xpad = (x.max() - x.min()) * 0.02
    xmin, xmax = x.min() - xpad, x.max() + xpad
    ax.axhspan(upper - loa_band_half_width, upper + loa_band_half_width,
               color="gray", alpha=0.25)
    ax.axhspan(lower - loa_band_half_width, lower + loa_band_half_width,
               color="gray", alpha=0.25)
    ax.axhline(md, color="black", lw=1)
    ax.axhline(upper, color="black", lw=0.8, linestyle="--")
    ax.axhline(lower, color="black", lw=0.8, linestyle="--")

    ax.text(xmax, upper, f"+1.96 s\n{upper:.1f}", fontsize=7,
            va="center", ha="right")
    ax.text(xmax, lower, f"-1.96 s\n{lower:.1f}", fontsize=7,
            va="center", ha="right")
    ax.text(xmax, md, f"{md:.1f}", fontsize=7, va="bottom", ha="right")
    add_annotation_box(ax, [f"Mean Difference +/- SD = {md:.1f} +/- {sd:.1f} mmHg"])

    ax.set_xlim(xmin, xmax)
    return finish_panel(ax, title, xlabel, ylabel, style)


def plot_histogram_panel(
    ax: plt.Axes,
    train_values: ArrayLike,
    test_values: ArrayLike,
    title: str,
    xlabel: str,
    train_color: str,
    style: FigureStyle | None = None,
    bins: int = 20,
    label_train: str = "Train",
    label_test: str = "Test",
) -> plt.Axes:
    """Draw one train/test histogram panel, such as (a) or (b).

    Args:
        ax: Matplotlib axis to draw on.
        train_values: Training values to histogram.
        test_values: Test values to histogram.
        title: Panel title.
        xlabel: x-axis label.
        train_color: Histogram color for training values.
        style: Optional shared style. Defaults to `FigureStyle()`.
        bins: Number of histogram bins.
        label_train: Legend label for training values.
        label_test: Legend label for test values.

    Returns:
        The Matplotlib axis containing the panel.
    """
    style = style or FigureStyle()
    train = np.asarray(train_values, dtype=float)
    test = np.asarray(test_values, dtype=float)

    ax.hist(train, bins=bins, color=train_color, edgecolor="black",
            alpha=0.7, label=label_train)
    ax.hist(test, bins=bins, color=style.test_hist_color, edgecolor="black",
            alpha=0.6, label=label_test)
    ax.legend(fontsize=7)

    train_mean, train_sd = mean_sd(train)
    test_mean, test_sd = mean_sd(test)
    add_annotation_box(
        ax,
        [f"{label_train}", f"mean = {train_mean:.2f}, SD = {train_sd:.2f}"],
        x=0.62,
        y=0.86,
        fontsize=7,
    )
    add_annotation_box(
        ax,
        [f"{label_test}", f"mean = {test_mean:.2f}, SD = {test_sd:.2f}"],
        x=0.62,
        y=0.56,
        fontsize=7,
    )
    return finish_panel(ax, title, xlabel, "Frequency", style)


def make_c_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """Create standalone panel (c): prediction vs reference correlation.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panel (c).
    """
    style = style or FigureStyle()
    fig, ax = plt.subplots(figsize=figsize)
    plot_correlation_panel(
        ax,
        reference=data.ref_train,
        prediction=data.pred_train,
        title="(c)",
        style=style,
    )
    fig.tight_layout()
    return fig


def make_d_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """Create standalone panel (d): calibration-reference Bland-Altman plot.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panel (d).
    """
    style = style or FigureStyle()
    fig, ax = plt.subplots(figsize=figsize)
    plot_bland_altman_panel(
        ax,
        x_values=(np.full_like(data.ref_train, data.ref_train.mean()) + data.ref_train) / 2,
        difference=data.calib_train - data.ref_train,
        title="(d)",
        xlabel="Mean of Population Average and Reference BP (mmHg)",
        ylabel="Calibration BP - Reference BP (mmHg)",
        style=style,
        point_color=style.prediction_color,
    )
    fig.tight_layout()
    return fig


def make_e_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """Create standalone panel (e): delta BP correlation plot.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panel (e).
    """
    style = style or FigureStyle()
    delta_ref = data.ref_test - data.calib_test
    delta_pred = data.pred_test - data.calib_test

    fig, ax = plt.subplots(figsize=figsize)
    plot_correlation_panel(
        ax,
        reference=delta_ref,
        prediction=delta_pred,
        title="(e)",
        xlabel="Reference BP - Calibration BP (mmHg)",
        ylabel="Cuffless BP - Calibration BP (mmHg)",
        style=style,
        point_color=style.calibration_color,
        show_identity=False,
        show_fit=True,
        open_markers=True,
    )
    fig.tight_layout()
    return fig


def make_f_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """Create standalone panel (f): calibration-reference Bland-Altman plot.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panel (f).
    """
    style = style or FigureStyle()
    fig, ax = plt.subplots(figsize=figsize)
    plot_bland_altman_panel(
        ax,
        x_values=(data.calib_test + data.ref_test) / 2,
        difference=data.calib_test - data.ref_test,
        title="(f)",
        xlabel="Mean of Calibration and Reference BP (mmHg)",
        ylabel="Calibration - Reference BP (mmHg)",
        style=style,
    )
    fig.tight_layout()
    return fig


def make_a_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (5.5, 4),
) -> plt.Figure:
    """Create standalone panel (a): absolute BP train/test histogram.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panel (a).
    """
    style = style or FigureStyle()
    fig, ax = plt.subplots(figsize=figsize)
    plot_histogram_panel(
        ax,
        train_values=data.ref_train,
        test_values=data.ref_test,
        title="(a)",
        xlabel="Systolic Blood Pressure (mmHg)",
        train_color=style.train_hist_color,
        style=style,
    )
    fig.tight_layout()
    return fig


def make_b_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (5.5, 4),
) -> plt.Figure:
    """Create standalone panel (b): BP change from calibration histogram.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panel (b).
    """
    style = style or FigureStyle()
    fig, ax = plt.subplots(figsize=figsize)
    plot_histogram_panel(
        ax,
        train_values=data.ref_train - data.calib_train,
        test_values=data.ref_test - data.calib_test,
        title="(b)",
        xlabel="BP Change from Calibration (mmHg)",
        train_color=style.change_hist_color,
        style=style,
    )
    fig.tight_layout()
    return fig


def make_cd_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (11, 4.5),
) -> plt.Figure:
    """Create legacy paired figure with panels (c) and (d).

    Prefer `make_c_figure` and `make_d_figure` when separate files are needed.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panels (c) and (d).
    """
    style = style or FigureStyle()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_correlation_panel(
        axes[0],
        reference=data.ref_train,
        prediction=data.pred_train,
        title="(c)",
        style=style,
    )
    plot_bland_altman_panel(
        axes[1],
        x_values=(np.full_like(data.ref_train, data.ref_train.mean()) + data.ref_train) / 2,
        difference=data.calib_train - data.ref_train,
        title="(d)",
        xlabel="Mean of Population Average and Reference BP (mmHg)",
        ylabel="Calibration BP - Reference BP (mmHg)",
        style=style,
        point_color=style.prediction_color,
    )
    fig.tight_layout()
    return fig


def make_ef_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (11, 4.5),
) -> plt.Figure:
    """Create legacy paired figure with panels (e) and (f).

    Prefer `make_e_figure` and `make_f_figure` when separate files are needed.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panels (e) and (f).
    """
    style = style or FigureStyle()
    delta_ref = data.ref_test - data.calib_test
    delta_pred = data.pred_test - data.calib_test

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_correlation_panel(
        axes[0],
        reference=delta_ref,
        prediction=delta_pred,
        title="(e)",
        xlabel="Reference BP - Calibration BP (mmHg)",
        ylabel="Cuffless BP - Calibration BP (mmHg)",
        style=style,
        point_color=style.calibration_color,
        show_identity=False,
        show_fit=True,
        open_markers=True,
    )
    plot_bland_altman_panel(
        axes[1],
        x_values=(data.calib_test + data.ref_test) / 2,
        difference=data.calib_test - data.ref_test,
        title="(f)",
        xlabel="Mean of Calibration and Reference BP (mmHg)",
        ylabel="Calibration - Reference BP (mmHg)",
        style=style,
    )
    fig.tight_layout()
    return fig


def make_ab_figure(
    data: BPCardData,
    style: FigureStyle | None = None,
    figsize: tuple[float, float] = (11, 4),
) -> plt.Figure:
    """Create legacy paired figure with histogram panels (a) and (b).

    Prefer `make_a_figure` and `make_b_figure` when separate files are needed.

    Args:
        data: BPCardData containing train/test arrays.
        style: Optional shared style. Defaults to `FigureStyle()`.
        figsize: Matplotlib figure size in inches.

    Returns:
        Matplotlib figure containing panels (a) and (b).
    """
    style = style or FigureStyle()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_histogram_panel(
        axes[0],
        train_values=data.ref_train,
        test_values=data.ref_test,
        title="(a)",
        xlabel="Systolic Blood Pressure (mmHg)",
        train_color=style.train_hist_color,
        style=style,
    )
    plot_histogram_panel(
        axes[1],
        train_values=data.ref_train - data.calib_train,
        test_values=data.ref_test - data.calib_test,
        title="(b)",
        xlabel="BP Change from Calibration (mmHg)",
        train_color=style.change_hist_color,
        style=style,
    )
    fig.tight_layout()
    return fig


def save_bp_card_figures(
    data: BPCardData,
    output_dir: str | Path = ".",
    dpi: int = 140,
    prefix: str = "fig",
    style: FigureStyle | None = None,
) -> dict[str, Path]:
    """Create and save each BP-Card panel as its own PNG figure.

    Args:
        data: BPCardData containing train/test arrays.
        output_dir: Directory where figures should be saved.
        dpi: Output image resolution.
        prefix: Filename prefix. Files are saved as `{prefix}_a.png` through
            `{prefix}_f.png`.
        style: Optional shared style. Defaults to `FigureStyle()`.

    Returns:
        Dictionary mapping panel labels `"a"` through `"f"` to saved paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {
        "a": make_a_figure(data, style=style),
        "b": make_b_figure(data, style=style),
        "c": make_c_figure(data, style=style),
        "d": make_d_figure(data, style=style),
        "e": make_e_figure(data, style=style),
        "f": make_f_figure(data, style=style),
    }

    saved_paths: dict[str, Path] = {}
    for name, fig in figures.items():
        path = output_path / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved_paths[name] = path
    return saved_paths


if __name__ == "__main__":
    demo_data = make_synthetic_bp_data()
    paths = save_bp_card_figures(demo_data)
    print("Saved:", ", ".join(str(path) for path in paths.values()))
