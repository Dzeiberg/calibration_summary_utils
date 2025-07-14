import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def plot_prior_distr(prior_summary_filepath: Path, figure_filepath: Path):
    with open(prior_summary_filepath, "r") as file:
        prior_summaries = json.load(file)
    fig, ax = plt.subplots(1, 1)
    sns.histplot(
        list(map(lambda d: d["mean"], prior_summaries.values())),
        bins=50,
        kde=False,
        ax=ax,
    )
    ax.set_xscale("log")
    fig.savefig(figure_filepath, bbox_inches="tight", dpi=300)
    print(f"Prior distribution plot saved to {figure_filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot prior distributions from a summary JSON file."
    )
    parser.add_argument(
        "prior_summary_filepath",
        type=Path,
        help="Filepath to the JSON file containing prior summaries.",
    )
    parser.add_argument(
        "figure_filepath",
        type=Path,
        help="Filepath to save the figure of prior distributions.",
    )

    args = parser.parse_args()

    plot_prior_distr(args.prior_summary_filepath, args.figure_filepath)
