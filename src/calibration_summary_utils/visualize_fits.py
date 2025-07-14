from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from assay_calibration.data_utils.dataset import Scoreset, BasicScoreset
from assay_calibration.fit_utils.fit import Fit
from typing import Optional, List

plt.set_loglevel("warning")  # or "error", "critical"


def visualize_dataset(
    summary_file: Path,
    fit_dicts: List[dict],
    save_filepath: Path | None = None,
    scoreset_filepath: Optional[Path] = None,
    scoreset: Optional[Scoreset | BasicScoreset] = None,
    **kwargs,
):
    if scoreset is None and scoreset_filepath is None:
        raise ValueError("Either scoreset or scoreset_filepath must be provided.")
    if scoreset is not None and scoreset_filepath is not None:
        raise ValueError(
            "Only one of scoreset or scoreset_filepath should be provided."
        )
    if scoreset_filepath is not None:
        scoreset_filepath = Path(scoreset_filepath)
        if not scoreset_filepath.exists():
            raise FileNotFoundError(f"The file {scoreset_filepath} does not exist.")
        # scoreset = Scoreset.from_json(scoreset_filepath)
        scoreset = Scoreset.from_json(
            scoreset_filepath,
            quantile_min=kwargs.get("quantile_min", 0.005),
            quantile_max=kwargs.get("quantile_max", 0.995),
        )
    else:
        if not (isinstance(scoreset, Scoreset) or isinstance(scoreset, BasicScoreset)):
            raise TypeError(
                "scoreset must be an instance of Scoreset or BasicScoreset."
            )
    summary_file = Path(summary_file)
    if not summary_file.exists():
        raise FileNotFoundError(f"The file {summary_file} does not exist.")
    with open(summary_file, "r") as file:
        summary = json.load(file)
    fits = [Fit.from_dict(scoreset, fit_dict) for fit_dict in fit_dicts]
    score_range = np.linspace(scoreset.scores.min(), scoreset.scores.max(), 1000)
    densities = np.stack(
        [
            [
                fit.model.get_sample_density(score_range, sample_num)  # type: ignore
                for fit in fits
            ]
            for sample_num in range(scoreset.n_samples)
        ]
    )
    fig, ax = plt.subplots(scoreset.n_samples, 1, figsize=(10, 6 * scoreset.n_samples))
    for sample_idx, (sample_scores, sample_name) in enumerate(scoreset.samples):
        sns.histplot(
            sample_scores,
            stat="density",
            ax=ax[sample_idx],
        )
        mean_density = np.mean(densities[sample_idx], axis=0)
        ax[sample_idx].plot(score_range, mean_density, color="red")
        ax[sample_idx].fill_between(
            score_range,
            *np.quantile(densities[sample_idx], [0.025, 0.975], axis=0),
            color="red",
            alpha=0.2,
            label="95% CI",
        )
        ax[sample_idx].set_title(f"{sample_name} Score Histogram and Mean Density")
        ax[sample_idx].set_xlabel("Score")
        ax[sample_idx].set_ylabel("Density")
        # ax[sample_idx].legend()
        linestyles = ["--", "-", "-.", ":", "-", "--"]
        for i, (thresholdP, thresholdB) in enumerate(
            zip(
                summary["final_pathogenic_thresholds"],
                summary["final_benign_thresholds"],
            )
        ):
            if not np.isnan(thresholdP):
                ax[sample_idx].axvline(thresholdP, color="red", linestyle=linestyles[i])
            if not np.isnan(thresholdB):
                ax[sample_idx].axvline(
                    thresholdB, color="blue", linestyle=linestyles[i]
                )
        if save_filepath is not None:
            fig.savefig(save_filepath, bbox_inches="tight", dpi=300)


def visualize_fits(
    scoresets_dir: Path, summaries_dir: Path, fits_file: Path, save_dir: Path
):
    """
    Visualizes the fits for each scoreset in the specified directory.

    Args:
        scoresets_dir (Path): Directory containing scoreset JSON files.
        summaries_dir (Path): Directory containing summary JSON files.
        fits_file (Path): File containing fit objects.
        save_dir (Path): Directory to save the visualizations.
    """
    scoresets_dir = Path(scoresets_dir)
    summaries_dir = Path(summaries_dir)
    fits_file = Path(fits_file)
    save_dir = Path(save_dir)

    if not scoresets_dir.exists():
        raise FileNotFoundError(f"The directory {scoresets_dir} does not exist.")
    if not summaries_dir.exists():
        raise FileNotFoundError(f"The directory {summaries_dir} does not exist.")
    if not fits_file.exists():
        raise FileNotFoundError(f"The file {fits_file} does not exist.")
    print("loading fits from", fits_file)
    with open(fits_file, "r") as file:
        fits = json.load(file)
    # Remove trailing underscores from each key in fits
    fits = {key.rstrip("_"): value for key, value in fits.items()}
    save_dir.mkdir(parents=True, exist_ok=True)

    def do_scoreset(scoreset_filepath):
        scoreset_name = scoreset_filepath.stem
        summary_filepath = summaries_dir / f"{scoreset_name}_summary.json"
        if not summary_filepath.exists():
            print(
                f"Summary file {summary_filepath} does not exist. Skipping {scoreset_filepath}."
            )
            return
        if scoreset_name not in fits:
            print(
                f"Fit for {scoreset_name} not found in {fits_file}. Skipping {scoreset_filepath}."
            )
            return
        print(f"Visualizing {scoreset_name}...")
        visualize_dataset(
            summary_filepath,
            fits[scoreset_name],
            save_filepath=save_dir / f"{scoreset_name}_fit_visualization.png",
            scoreset_filepath=scoreset_filepath,
        )

    scoreset_filepaths = list(scoresets_dir.glob("*.json"))
    for scoreset_filepath in tqdm(scoreset_filepaths, desc="Processing scoresets"):
        do_scoreset(scoreset_filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize fits for scoresets.")
    parser.add_argument(
        "scoresets_dir", type=Path, help="Directory containing scoreset JSON files."
    )
    parser.add_argument(
        "summaries_dir", type=Path, help="Directory containing summary JSON files."
    )
    parser.add_argument("fits_file", type=Path, help="File containing fit objects.")
    parser.add_argument(
        "save_dir", type=Path, help="Directory to save the visualizations."
    )

    args = parser.parse_args()

    visualize_fits(
        args.scoresets_dir, args.summaries_dir, args.fits_file, args.save_dir
    )
    print(f"Visualizations saved to {args.save_dir}")
