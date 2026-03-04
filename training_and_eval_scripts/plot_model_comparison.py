"""
Compare model performance distributions from per-image metrics files.

Each model points to a `metrics_per_image.csv` (or a directory containing it).
The script plots Precision, Recall, and F1 distributions with:
- boxplots (distribution summary)
- jittered dots (individual image-level values)

Config usage:
- Copy `training_and_eval_scripts/configs/plot_model_comparison_config_template.toml` to
  `training_and_eval_scripts/configs/plot_model_comparison_config_local.toml`.
- Edit `_local.toml` to your preferred settings and run the script.
- If `_local.toml` is missing, the script falls back to `_template.toml`.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.io_helpers import load_script_config, normalize_user_path, require_file

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "plot_model_comparison_config", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

models_cfg = cfg["models"]
if not isinstance(models_cfg, list) or len(models_cfg) == 0:
    raise ValueError("models must be a non-empty list of tables in config.")

metric_type = str(cfg["metric_type"]).strip().lower()
if metric_type not in {"centroid", "iou"}:
    raise ValueError("metric_type must be either 'centroid' or 'iou'.")

output_dir = normalize_user_path(cfg["output_dir"])
output_prefix = str(cfg["output_prefix"])
file_type = str(cfg["file_type"])
show_plot = bool(cfg["show_plot"])
y_min = float(cfg["y_min"])
y_max = float(cfg["y_max"])
point_alpha = float(cfg["point_alpha"])
point_size = float(cfg["point_size"])
jitter = float(cfg["jitter"])
show_median_labels = bool(cfg.get("show_median_labels", cfg.get("show_mean_labels", False)))
median_label_decimals = int(cfg.get("median_label_decimals", cfg.get("mean_label_decimals", 3)))
show_overall_markers = bool(cfg["show_overall_markers"])
show_overall_value_labels = bool(cfg.get("show_overall_value_labels", False))
overall_label_decimals = int(cfg.get("overall_label_decimals", 3))

if y_min >= y_max:
    raise ValueError("y_min must be smaller than y_max.")
if jitter < 0:
    raise ValueError("jitter must be >= 0.")
if median_label_decimals < 0:
    raise ValueError("median_label_decimals must be >= 0.")
if overall_label_decimals < 0:
    raise ValueError("overall_label_decimals must be >= 0.")

if metric_type == "centroid":
    precision_col = "Precision_centroid"
    recall_col = "Recall_centroid"
    f1_col = "F1_centroid"
else:
    precision_col = "Precision_iouMatch"
    recall_col = "Recall_iouMatch"
    f1_col = "F1_iouMatch"

metric_columns = {
    "Precision": precision_col,
    "Recall": recall_col,
    "F1": f1_col,
}


def resolve_metrics_csv(path_like: str) -> Path:
    """Return a valid metrics_per_image.csv path from CSV or directory input."""
    p = normalize_user_path(path_like)
    if p.is_dir():
        p = p / "metrics_per_image.csv"
    return require_file(p, "metrics_per_image.csv")


def path_basename_key(p: str | Path) -> str:
    """Return normalized last path segment (folder/file name) for matching."""
    s = str(p).replace("\\", "/").rstrip("/")
    if not s:
        return ""
    return s.split("/")[-1].lower()


def lookup_overall_metrics(
    eval_log_path: str | Path, expected_out_dir: Path, metric_family: str
) -> dict[str, float]:
    """Fetch overall Precision/Recall/F1 by matching out_dir basename in evaluation_log.csv."""
    eval_csv = require_file(normalize_user_path(eval_log_path), "evaluation_log.csv")
    eval_df = pd.read_csv(eval_csv)

    if "out_dir" not in eval_df.columns:
        raise ValueError(f"Missing column 'out_dir' in: {eval_csv}")

    expected_name = path_basename_key(expected_out_dir)
    out_dir_names = eval_df["out_dir"].astype(str).map(path_basename_key)
    rows = eval_df[out_dir_names == expected_name]

    if rows.empty:
        raise ValueError(
            f"No row in {eval_csv} where out_dir basename matches '{expected_name}'.\n"
            f"Expected out_dir folder: {expected_out_dir}"
        )

    row = rows.iloc[-1]
    p_col = f"overall_precision_{metric_family}"
    r_col = f"overall_recall_{metric_family}"
    f_col = f"overall_f1_{metric_family}"

    values = {
        "Precision": float(pd.to_numeric(row[p_col], errors="coerce")),
        "Recall": float(pd.to_numeric(row[r_col], errors="coerce")),
        "F1": float(pd.to_numeric(row[f_col], errors="coerce")),
    }

    return values


# -------------------------
# LOAD DATA
# -------------------------
records = []
summary_rows = []
overall_markers = {}

for model_entry in models_cfg:
    if not isinstance(model_entry, dict):
        raise ValueError("Each entry in models must be a table with label and metrics_path.")

    label = model_entry.get("label")
    metrics_path_raw = model_entry.get("metrics_path")

    if not label or not metrics_path_raw:
        raise ValueError("Each model entry must contain non-empty 'label' and 'metrics_path'.")

    metrics_path = resolve_metrics_csv(metrics_path_raw)
    model_out_dir = metrics_path.parent
    df = pd.read_csv(metrics_path)

    missing_cols = set(metric_columns.values()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"File missing required columns for metric_type='{metric_type}': {metrics_path}\n"
            + ", ".join(sorted(missing_cols))
        )

    # Coerce numeric and drop bad rows so mixed-content files are still usable.
    precision_vals = pd.to_numeric(df[precision_col], errors="coerce").dropna().to_numpy()
    recall_vals = pd.to_numeric(df[recall_col], errors="coerce").dropna().to_numpy()
    f1_vals = pd.to_numeric(df[f1_col], errors="coerce").dropna().to_numpy()

    if len(precision_vals) == 0 or len(recall_vals) == 0 or len(f1_vals) == 0:
        raise ValueError(f"No valid numeric metric values found in: {metrics_path}")

    for metric_name, values in [
        ("Precision", precision_vals),
        ("Recall", recall_vals),
        ("F1", f1_vals),
    ]:
        for v in values:
            records.append({"Model": str(label), "Metric": metric_name, "Value": float(v)})

    summary_rows.append(
        {
            "Model": str(label),
            "N_images_precision": int(len(precision_vals)),
            "N_images_recall": int(len(recall_vals)),
            "N_images_f1": int(len(f1_vals)),
            "Mean_precision": float(np.mean(precision_vals)),
            "Mean_recall": float(np.mean(recall_vals)),
            "Mean_f1": float(np.mean(f1_vals)),
            "Std_precision": float(np.std(precision_vals, ddof=1)) if len(precision_vals) > 1 else 0.0,
            "Std_recall": float(np.std(recall_vals, ddof=1)) if len(recall_vals) > 1 else 0.0,
            "Std_f1": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
            "metrics_path": str(metrics_path),
        }
    )

    if show_overall_markers:
        eval_log_path = model_entry.get("evaluation_log_path")
        if not eval_log_path:
            raise ValueError(
                f"Model '{label}' is missing 'evaluation_log_path' but "
                "show_overall_markers=true."
            )
        overall_vals = lookup_overall_metrics(eval_log_path, model_out_dir, metric_type)
        for metric_name, val in overall_vals.items():
            overall_markers[(str(label), metric_name)] = val

plot_df = pd.DataFrame(records)
if plot_df.empty:
    raise ValueError("No data found to plot.")

# Keep user-specified model order from config
model_order = [str(m["label"]) for m in models_cfg]
metric_order = ["Precision", "Recall", "F1"]

output_dir.mkdir(parents=True, exist_ok=True)

summary_df = pd.DataFrame(summary_rows)
summary_csv = output_dir / f"{output_prefix}_{metric_type}_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Saved summary CSV to: {summary_csv}")

# -------------------------
# PLOT (boxplot + individual dots)
# -------------------------
plt.figure(figsize=(12, 7))
ax = plt.gca()

x_centers = np.arange(len(metric_order), dtype=float)
n_models = len(model_order)
box_width = 0.75 / max(n_models, 1)
colors = plt.cm.get_cmap("tab10", n_models)

for i, model_name in enumerate(model_order):
    color = colors(i)
    offset = (i - (n_models - 1) / 2) * box_width

    for j, metric_name in enumerate(metric_order):
        vals = plot_df[
            (plot_df["Model"] == model_name) & (plot_df["Metric"] == metric_name)
        ]["Value"].to_numpy()

        x_pos = x_centers[j] + offset

        # Boxplot (distribution summary)
        ax.boxplot(
            vals,
            positions=[x_pos],
            widths=box_width * 0.85,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )

        # Jittered individual points (per-image values)
        x_jitter = np.random.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_pos) + x_jitter,
            vals,
            s=point_size,
            alpha=point_alpha,
            color=color,
            edgecolors="black",
            linewidths=0.3,
        )

        if show_median_labels:
            median_val = float(np.median(vals))
            ax.annotate(
                f"{median_val:.{median_label_decimals}f}",
                xy=(x_pos, median_val),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color="black",
            )

        if show_overall_markers:
            overall_val = overall_markers[(model_name, metric_name)]
            line_half_width = box_width * 0.34
            ax.plot(
                [x_pos - line_half_width, x_pos + line_half_width],
                [overall_val, overall_val],
                color="black",
                linewidth=1.6,
                linestyle=":",
                zorder=4,
            )
            if show_overall_value_labels:
                ax.annotate(
                    f"{overall_val:.{overall_label_decimals}f}",
                    xy=(x_pos, overall_val),
                    xytext=(6, -8),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color="black",
                )

# Manual legend handles (one color per model)
handles = [
    plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=colors(i),
               markeredgecolor="black", markersize=7)
    for i, name in enumerate(model_order)
]

ax.set_xticks(x_centers)
ax.set_xticklabels(metric_order)
ax.set_ylabel(f"Per-image {metric_type.title()} score")
ax.set_title(f"Model comparison from metrics_per_image.csv ({metric_type})")
# Add a little visual headroom above the top tick without introducing a higher tick.
y_span = y_max - y_min
ax.set_ylim(y_min, y_max + (0.06 * y_span))
ax.set_yticks(np.linspace(y_min, y_max, 6))
ax.grid(axis="y", alpha=0.3)
ax.legend(handles=handles, title="Model")
plt.tight_layout()

plot_path = output_dir / f"{output_prefix}_{metric_type}.{file_type}"
plt.savefig(plot_path)
print(f"Saved comparison plot to: {plot_path}")

if show_plot:
    plt.show()
