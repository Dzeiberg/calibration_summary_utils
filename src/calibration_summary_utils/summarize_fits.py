import pandas as pd
from pathlib import Path
from assay_calibration.fit_utils.fit import Fit
from assay_calibration.data_utils.dataset import Scoreset, BasicScoreset
import json
import numpy as np
from tqdm import tqdm
from typing import List
from .summary_utils import get_priors, get_thresholds, summarize_thresholds
from .select_final_models import aggregate_and_select_final_models
from .visualize_fits import visualize_fits, visualize_dataset


def is_inverted(scoreset):
    pathogenic_scores = scoreset.scores[scoreset.sample_assignments[:, 0] == 1]
    benign_scores = scoreset.scores[scoreset.sample_assignments[:, 1] == 1]
    return np.median(pathogenic_scores) > np.median(benign_scores)


def summarize_pillar_fits(
    models_dir: Path,
    scoresets_dir: Path,
    final_models_filepath: Path,
    summaries_dir: Path,
    figures_dir: Path,
    dataframe_filepath: Path,
    **kwargs,
):
    """
    Summarize the fits from the models and scoresets.
    Args:
        models_dir (Path): Directory containing the models JSON files.
        scoresets_dir (Path): Directory containing the scoresets.
        final_models_filepath (Path): Path to which the final models file will be saved.
        summaries_dir (Path): Path to the output summary JSON file.
        figures_dir (Path): Directory to save the fit visualizations.
        dataframe_filepath (Path): Path to the pillar dataframe CSV file.

    Optional Args:
        - final_threshold_quantile (float): The quantile to use for final thresholds. Default is 0.05
    """
    dataframe_filepath = Path(dataframe_filepath)
    if not dataframe_filepath.exists():
        raise FileNotFoundError(f"The file {dataframe_filepath} does not exist.")
    if not dataframe_filepath.is_file():
        raise ValueError(f"The path {dataframe_filepath} is not a file.")
    print(f"Loading pillar dataframe from {dataframe_filepath}")
    # Load the pillar dataframe
    pillar_df = pd.read_csv(dataframe_filepath)

    final_threshold_quantile = kwargs.get("final_threshold_quantile", 0.05)
    models_dir = Path(models_dir)
    scoresets_dir = Path(scoresets_dir)
    summaries_dir = Path(summaries_dir)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    if not models_dir.exists():
        raise FileNotFoundError(f"The file {models_dir} does not exist.")
    final_models_filepath = Path(final_models_filepath)
    final_models_filepath.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Aggregating and selecting final models from {models_dir} to {final_models_filepath}"
    )
    aggregate_and_select_final_models(
        models_dir, final_models_filepath, num_models=None
    )
    if not scoresets_dir.exists():
        raise FileNotFoundError(f"The directory {scoresets_dir} does not exist.")
    with open(final_models_filepath, "r") as file:
        models = json.load(file)
    for scoreset_name, fit_dicts in tqdm(
        models.items(), desc="Summarizing scoresets", total=len(models)
    ):
        if scoreset_name[-1] == "_":
            scoreset_name = scoreset_name[:-1]
        scoreset_summary_file = summaries_dir / f"{scoreset_name}_summary.json"
        if scoreset_summary_file.exists():
            print(f"Summary for {scoreset_name} already exists. Skipping...")
            continue
        subset = pillar_df[pillar_df.Dataset == scoreset_name].drop_duplicates(
            subset=["ID"]
        )
        if subset.empty:
            raise ValueError(
                f"No data found for scoreset {scoreset_name} in the dataframe."
            )
        # score_values = pd.to_numeric(subset['auth_reported_score'].values, errors='coerce')
        scoreset_result = {
            "scoreset_name": scoreset_name,
            "calibration_method": "Multi-sample skew normal mixture model",
        }
        scoreset_path = scoresets_dir / f"{scoreset_name}.json"
        if not scoreset_path.exists():
            raise FileNotFoundError(f"The scoreset {scoreset_path} does not exist.")
        scoreset = Scoreset.from_json(scoreset_path)
        fits = [Fit.from_dict(scoreset, fit_data) for fit_data in fit_dicts]
        priors = get_priors(scoreset, fits)
        scoreset_result["priors"] = priors
        median_prior = np.median(priors)
        scoreset_result["median_prior"] = median_prior
        inverted = is_inverted(scoreset)
        pathogenic_threshold_sets, benign_threshold_sets = get_thresholds(
            fits,
            median_prior,
            inverted,
        )
        scoreset_result[
            "pathogenic_threshold_sets"
        ] = pathogenic_threshold_sets.tolist()
        scoreset_result["benign_threshold_sets"] = benign_threshold_sets.tolist()
        scoreset_result["inverted"] = "inverted" if inverted else "canonical"
        final_pathogenic_thresholds, final_benign_thresholds = summarize_thresholds(
            pathogenic_threshold_sets,
            benign_threshold_sets,
            final_threshold_quantile,
            bool(inverted),
        )
        scoreset_result[
            "final_pathogenic_thresholds"
        ] = final_pathogenic_thresholds.tolist()
        scoreset_result["final_benign_thresholds"] = final_benign_thresholds.tolist()
        scoreset_result["final_threshold_quantile"] = final_threshold_quantile
        with open(scoreset_summary_file, "w") as f:
            json.dump(scoreset_result, f, indent=4)
        print(f"Summary written to {scoreset_summary_file}")
    print(f"Final models saved to {final_models_filepath}")
    print(f"Summaries saved to {summaries_dir}")
    print(f"Visualizing fits and saving to {figures_dir}")
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    visualize_fits(
        scoresets_dir=scoresets_dir,
        summaries_dir=summaries_dir,
        fits_file=final_models_filepath,
        save_dir=figures_dir,
    )


def summarize_fits(
    fits: List[Fit],
    scoreset: Scoreset | BasicScoreset,
    summary_file_savepath: str | Path,
    figure_savepath: str | Path,
    **kwargs,
):
    """
    Summarize the fits and save the summary and figure.
    Args:
        fits (List[Fit]): List of Fit objects to summarize.
        scoreset (Scoreset|BasicScoreset): The scoreset associated with the fits.
        summary_file_savepath (str|Path): Path to save the summary JSON file.
        figure_savepath (str|Path): Path to save the figure.

    Optional Args:
        - final_threshold_quantile (float): The quantile to use for final thresholds. Default is 0.05
    """
    scoreset_result = {}
    priors = get_priors(scoreset, fits)
    scoreset_result["priors"] = priors
    median_prior = np.median(priors)
    scoreset_result["median_prior"] = median_prior
    inverted = is_inverted(scoreset)
    pathogenic_threshold_sets, benign_threshold_sets = get_thresholds(
        fits,
        median_prior,
        inverted,
    )
    scoreset_result["pathogenic_threshold_sets"] = pathogenic_threshold_sets.tolist()
    scoreset_result["benign_threshold_sets"] = benign_threshold_sets.tolist()
    scoreset_result["inverted"] = "inverted" if inverted else "canonical"
    final_threshold_quantile = kwargs.get("final_threshold_quantile", 0.05)
    if final_threshold_quantile < 0 or final_threshold_quantile > 1:
        raise ValueError(
            f"final_threshold_quantile must be between 0 and 1, got {final_threshold_quantile}"
        )
    final_pathogenic_thresholds, final_benign_thresholds = summarize_thresholds(
        pathogenic_threshold_sets,
        benign_threshold_sets,
        final_threshold_quantile,
        inverted,
    )
    scoreset_result[
        "final_pathogenic_thresholds"
    ] = final_pathogenic_thresholds.tolist()
    scoreset_result["final_benign_thresholds"] = final_benign_thresholds.tolist()
    scoreset_result["final_threshold_quantile"] = final_threshold_quantile
    with open(summary_file_savepath, "w") as f:
        json.dump(scoreset_result, f, indent=4)
    print(f"Summary written to {summary_file_savepath}")
    figure_savepath = Path(figure_savepath)
    figure_savepath.parent.mkdir(parents=True, exist_ok=True)
    visualize_dataset(
        Path(summary_file_savepath),
        [fit.to_dict() for fit in fits],
        save_filepath=figure_savepath,
        scoreset=scoreset,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize fits from models and scoresets."
    )
    parser.add_argument("models_dir", type=Path, help="Path to the models JSON file.")
    parser.add_argument(
        "scoresets_dir", type=Path, help="Directory containing the scoresets."
    )
    parser.add_argument(
        "final_models_filepath",
        type=Path,
        help="Path to which the final models file will be saved.",
    )
    parser.add_argument(
        "summaries_dir", type=Path, help="Output directory for the fit summaries."
    )
    parser.add_argument(
        "figures_dir", type=Path, help="Output directory for the fit visualizations."
    )
    parser.add_argument(
        "dataframe_filepath", type=Path, help="Path to the pillar dataframe CSV file."
    )
    parser.add_argument(
        "--final_threshold_quantile",
        type=float,
        default=0.05,
        help="Quantile for final thresholds (default: 0.05)",
    )
    args = parser.parse_args()

    summarize_pillar_fits(
        models_dir=args.models_dir,
        scoresets_dir=args.scoresets_dir,
        final_models_filepath=args.final_models_filepath,
        summaries_dir=args.summaries_dir,
        figures_dir=args.figures_dir,
        dataframe_filepath=args.dataframe_filepath,
        final_threshold_quantile=args.final_threshold_quantile,
    )
    # Example usage:
    # from calibration_summary_utils.summarize_fits import summarize_fits
    #
    # summarize_fits(
    #     models_dir="/path/to/models",
    #     scoresets_dir="/path/to/scoresets",
    #     final_models_filepath="/path/to/save/final_models.json",
    #     summaries_dir="/path/to/save/summaries",
    #     figures_dir="/path/to/save/figures",
    #     dataframe_filepath="/path/to/pillar_dataframe.csv",
    #     final_threshold_quantile=0.05
    # )
