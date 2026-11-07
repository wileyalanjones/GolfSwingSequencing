"""Visualization utilities for training metrics and PCE results."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SEQUENCES = [
    "Total",
    "Address",
    "Toe-up",
    "Mid-backswing",
    "Top",
    "Mid-downswing",
    "Impact",
    "Mid-follow-through",
    "Finish",
]

# Reported probabilistic fraction of correct events (PCE), per sequence.
PCE = [
    0.5932142857142857,
    0.08857143,
    0.66285714,
    0.64857143,
    0.53142857,
    0.86,
    0.89428571,
    0.91714286,
    0.14285714,
]

# Metric log file -> display label used in the plots.
_TRAINING_LOGS = {
    "Clip_Length_16": "M256_B1_C16.csv",
    "Clip_Length_32": "CL32_U256_B10.csv",
    "Clip_Length_64": "M256_B5.csv",
}


def load_training_results(results_dir=None) -> pd.DataFrame:
    """Load and align the three training metric logs by epoch."""
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR

    frames = []
    for label, fname in _TRAINING_LOGS.items():
        df = pd.read_csv(results_dir / fname)
        df = df.iloc[-50:].copy()
        df["Model"] = label
        df["epoch"] = range(1, len(df) + 1)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_pce_dataframe() -> pd.DataFrame:
    """Build the per-sequence PCE results frame."""
    return pd.DataFrame({"Sequence": SEQUENCES, "PCE": PCE})


def plot_metric(final_df, column, title, ylabel=None, ylim=None):
    """Line plot of a single validation metric across epochs, per model."""
    ax = sns.lineplot(data=final_df, x="epoch", y=column, hue="Model", legend="brief")
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.legend(fontsize="small")
    plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)
    return ax


def plot_pce(pce_df):
    """Bar plot of PCE per swing sequence."""
    ax = sns.barplot(pce_df, x="PCE", y="Sequence", hue="PCE")
    for bars in ax.containers:
        if isinstance(bars, BarContainer):
            ax.bar_label(bars, fmt="%.2f")
    ax.margins(x=0.1)
    plt.title("PCE Results")
    return ax


def main() -> None:
    """Render the full set of result plots."""
    final_df = load_training_results()
    pce_df = build_pce_dataframe()

    plot_metric(final_df, "val_event_auc_pr", "PR AUC per Model Tested", ylim=(0, 0.35))
    plt.show()
    plot_metric(final_df, "val_frame_accuracy", "Frame Accuracy per Model Tested")
    plt.show()
    plot_metric(final_df, "val_event_precision", "Event Precision per Model Tested")
    plt.show()
    plot_metric(final_df, "val_event_recall", "Event Recall per Model Tested")
    plt.show()
    plot_pce(pce_df)
    plt.show()


if __name__ == "__main__":
    main()