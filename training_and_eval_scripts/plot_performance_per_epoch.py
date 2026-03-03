"""
Plot overall F1 centroid score as a function of training epochs.

Reads an evaluation log CSV, selects one base model plus any entries with suffix
`_epoch_XXXX`, and plots performance metrics against epoch number.

"""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.io_helpers import load_script_config, normalize_user_path, require_file

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
test_mode = False
cfg = load_script_config(
    Path(__file__), "plot_performance_per_epoch_config", test_mode=test_mode
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------
evaluation_log_path = require_file(
    normalize_user_path(cfg["evaluation_log_path"]), "Evaluation log CSV"
)

# Base model name as written in column "model" for the final model
model_name = cfg["model_name"]

# Number of epochs for the base model entry (without _epoch_XXXX suffix)
base_model_epochs = cfg["base_model_epochs"]

# Metric family to plot: "centroid" (default) or "iou"
metric_type = cfg.get("metric_type", "centroid").strip().lower()

# Plot and output settings
line_color = cfg["line_color"]
marker_style = cfg["marker_style"]
file_type = cfg["file_type"]
show_plot = cfg["show_plot"]
y_min = cfg["y_min"]
y_max = cfg["y_max"]


def get_epoch_for_model(logged_model_name: str, base_name: str, base_epochs: int):
    """Return epoch number for base model or _epoch_XXXX variant, otherwise None."""
    if logged_model_name == base_name:
        return int(base_epochs)

    pattern = re.escape(base_name) + r"_epoch_(\d{4})$"
    match = re.match(pattern, logged_model_name)
    if match:
        return int(match.group(1))

    return None


### MAIN CODE
if base_model_epochs < 0:
    raise ValueError("base_model_epochs must be >= 0.")
if y_min >= y_max:
    raise ValueError("y_min must be smaller than y_max.")
if metric_type not in {"centroid", "iou"}:
    raise ValueError("metric_type must be either 'centroid' or 'iou'.")

f1_col = f"overall_f1_{metric_type}"
recall_col = f"overall_recall_{metric_type}"
precision_col = f"overall_precision_{metric_type}"

eval_df = pd.read_csv(evaluation_log_path)

required_columns = {
    "model",
    f1_col,
    recall_col,
    precision_col,
}
missing_cols = required_columns - set(eval_df.columns)
if missing_cols:
    raise ValueError(
        "evaluation_log.csv is missing required columns: "
        + ", ".join(sorted(missing_cols))
    )

# Keep only rows for the selected model family
model_rows = eval_df.copy()
model_rows["epoch"] = model_rows["model"].astype(str).map(
    lambda x: get_epoch_for_model(x, model_name, base_model_epochs)
)
model_rows = model_rows.dropna(subset=["epoch"]).copy()

if model_rows.empty:
    raise ValueError(
        f'No rows found for model "{model_name}" or its _epoch_XXXX variants.'
    )

model_rows["epoch"] = model_rows["epoch"].astype(int)
model_rows[f1_col] = pd.to_numeric(
    model_rows[f1_col], errors="coerce"
)
model_rows[recall_col] = pd.to_numeric(
    model_rows[recall_col], errors="coerce"
)
model_rows[precision_col] = pd.to_numeric(
    model_rows[precision_col], errors="coerce"
)
model_rows = model_rows.dropna(
    subset=[
        f1_col,
        recall_col,
        precision_col,
    ]
)

if model_rows.empty:
    raise ValueError(
        f"No valid numeric values found in {f1_col}, "
        f"{recall_col}, and {precision_col}."
    )

# If duplicate epochs exist, keep the last occurrence in the CSV
model_rows = model_rows.drop_duplicates(subset=["epoch"], keep="last")
model_rows = model_rows.sort_values("epoch")

out_dir = evaluation_log_path.parent / "f_score_eval"
out_path = out_dir / f"{model_name}_performance_per_epoch_{metric_type}.{file_type}"

plt.figure(figsize=(10, 10))
plt.plot(
    model_rows["epoch"],
    model_rows[f1_col],
    marker=marker_style,
    linestyle="-",
    color=line_color,
    label=f"Overall F1 ({metric_type})",
)
plt.plot(
    model_rows["epoch"],
    model_rows[recall_col],
    marker=marker_style,
    linestyle="-",
    color="green",
    label=f"Overall Recall ({metric_type})",
)
plt.plot(
    model_rows["epoch"],
    model_rows[precision_col],
    marker=marker_style,
    linestyle="-",
    color="orange",
    label=f"Overall Precision ({metric_type})",
)

# Set x-axis from 0 to base_model_epochs + 100 and label each data-point epoch.
ax = plt.gca()
min_epoch = int(model_rows["epoch"].min())
x_axis_max = int(base_model_epochs) + 100
ax.set_xlim(0, x_axis_max)
data_epochs = model_rows["epoch"].astype(int).tolist()
xticks = sorted(set([0, min_epoch, x_axis_max] + data_epochs))
ax.set_xticks(xticks)

# Annotate highest and lowest overall F1 points
f1_max_idx = model_rows[f1_col].idxmax()
f1_min_idx = model_rows[f1_col].idxmin()

max_x = model_rows.loc[f1_max_idx, "epoch"]
max_y = model_rows.loc[f1_max_idx, f1_col]
min_x = model_rows.loc[f1_min_idx, "epoch"]
min_y = model_rows.loc[f1_min_idx, f1_col]

if f1_max_idx == f1_min_idx:
    plt.annotate(
        f"Max/Min F1: {max_y:.3f}",
        xy=(max_x, max_y),
        xytext=(12, 12),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="black", lw=1),
    )
else:
    plt.annotate(
        f"Max F1: {max_y:.3f}",
        xy=(max_x, max_y),
        xytext=(12, 12),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="black", lw=1),
    )
    plt.annotate(
        f"Min F1: {min_y:.3f}",
        xy=(min_x, min_y),
        xytext=(12, -16),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="black", lw=1),
    )

plt.title(f"Performance vs Epochs: {model_name}")
plt.xlabel("Epochs")
plt.ylabel(f"Overall Metrics ({metric_type})")
plt.ylim(y_min, y_max)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(out_path)
print(f"Saved plot to: {out_path}")

if show_plot:
    plt.show()
