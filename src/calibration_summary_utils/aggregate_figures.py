import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from tqdm import tqdm


def summarize_all(
    summaries_dir: Path,
    output_summary_file: Path | None = None,
    output_image_file: Path | None = None,
):
    """
    Visualize the summary of fit results from a JSON file.

    Args:
        summary_dir (Path): The directory containing the JSON files summarizing the final calibration for each scoreset.
        output_summary_file (Path|None): Optional path to save the summary DataFrame as an Excel file.
        output_image_file (Path|None): Optional path to save the heatmap image.
    """
    # Load the summary data
    max_points = [0, 1, 2, 3, 4, 8]
    summaries_dir = Path(summaries_dir)
    summaries = {}
    if not summaries_dir.exists():
        raise FileNotFoundError(f"The directory {summaries_dir} does not exist.")
    summary_files = list(summaries_dir.glob("*.json"))
    if not summary_files:
        raise FileNotFoundError(
            f"No JSON files found in the directory {summaries_dir}."
        )
    for file in tqdm(summary_files):
        with open(file, "r") as f:
            summary = json.load(f)
        max_points_benign = (
            -1 * max_points[5 - np.isnan(summary["final_benign_thresholds"]).sum()]
        )
        max_points_pathogenic = max_points[
            5 - np.isnan(summary["final_pathogenic_thresholds"]).sum()
        ]
        P = np.array(summary["pathogenic_threshold_sets"])
        B = np.array(summary["benign_threshold_sets"])
        summaries[file.stem] = {
            "max_points_benign": max_points_benign,
            "max_points_pathogenic": max_points_pathogenic,
            "median_prior": np.median(summary["priors"]),
            "frac_inf_P": np.isinf(P).sum(0) / P.shape[0],
            "frac_inf_B": np.isinf(B).sum(0) / B.shape[0],
        }

    # Create a pivot table with counts using max_points_benign and max_points_pathogenic
    summaries = pd.DataFrame.from_dict(summaries, orient="index")
    summaries.sort_index(inplace=True)
    if output_summary_file is not None:
        summaries.to_excel(
            output_summary_file, index_label="Dataset", sheet_name="Calibration Summary"
        )
    pivot_table = summaries.pivot_table(
        index="max_points_benign",
        columns="max_points_pathogenic",
        aggfunc="size",
        fill_value=0,
    )
    # Sort the pivot table rows and columns in ascending order
    pivot_table = pivot_table.sort_index(axis=0, ascending=True).sort_index(axis=1)
    # Plot the pivot table as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_table, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Count"}
    )
    plt.title("Heatmap of Counts")
    plt.xlabel("Max Points Pathogenic")
    plt.ylabel("Max Points Benign")
    if output_image_file is not None:
        plt.savefig(output_image_file, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize summary of fit results from a JSON file."
    )
    parser.add_argument(
        "summaries_dir", type=Path, help="Path to the summary JSON file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for the summary (optional).",
    )
    parser.add_argument(
        "--output_image_file",
        type=Path,
        default=None,
        help="Output file for the visualization (optional).",
    )
    args = parser.parse_args()

    fig = summarize_all(args.summaries_dir, args.output, args.output_image_file)
