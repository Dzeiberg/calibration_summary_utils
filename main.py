from pathlib import Path
from tqdm import tqdm
from src.calibration_summary_utils.scoreset import Scoreset
from assay_calibration.data_utils.dataset import BasicScoreset
from assay_calibration.fit_utils.fit import Fit
from src.calibration_summary_utils.summarize_fits import summarize_fits


def main(scoreset_filepath: Path, summary_filepath: Path, fig_filepath: Path, **kwargs):
    """
    Main function to process a scoreset, perform fits, and generate a summary and figure.
    Args:
        scoreset_filepath (Path): Path to the input scoreset file. 
                                  The format depends on the `scoreset_type` parameter.
        summary_filepath (Path): Path to save the summary file.
        fig_filepath (Path): Path to save the generated figure.
        **kwargs: Additional optional parameters:
            - num_iterations (int): Number of iterations to perform. Default is 10.
            - num_fits (int): Number of fits to perform for each iteration. Default is 10.
            - core_limit (int): Maximum number of cores to use. Default is 1.
            - component_range (list): Range of components to consider. Default is [2, 3].
            - scoreset_type (str): Type of scoreset. Must be either "BasicScoreset" or "PillarProject". 
                                   Default is "BasicScoreset".
            - bootstrap (bool): Whether to use bootstrapping, evaluating on the out-of-bag samples,
                                or to fit on the entire dataset and use the with with the maximum
                                train likelihood. Default is True
            - summarize (bool): Whether to run the summary and save results. Default is False.
    Raises:
        ValueError: If an unknown `scoreset_type` is provided.
    Side Effects:
        - Saves a summary file to `summary_filepath`.
        - Saves a figure to `fig_filepath`.
        - Prints the paths to the saved summary and figure.
    Example:
        main(
            scoreset_filepath=Path("input.csv"),
            summary_filepath=Path("summary.txt"),
            fig_filepath=Path("figure.png"),
            num_fits=10,
            core_limit=4,
            component_range=[2, 3],
            scoreset_type="BasicScoreset",
            bootstrap=True,
            summarize=True
        )
    """
    summary_filepath = Path(summary_filepath)
    fig_filepath = Path(fig_filepath)
    summary_filepath.parent.mkdir(parents=True, exist_ok=True)
    fig_filepath.parent.mkdir(parents=True, exist_ok=True)
    num_fits = kwargs.get("num_fits", 10)
    core_limit = kwargs.get("core_limit", 1)
    component_range = kwargs.get("component_range", [2, 3])
    scoreset_type = kwargs.get("scoreset_type", "BasicScoreset")
    if scoreset_type == "BasicScoreset":
        scoreset = BasicScoreset.from_csv(scoreset_filepath)
    elif scoreset_type == "PillarProject":
        scoreset = Scoreset.from_json(scoreset_filepath)
    else:
        raise ValueError(f"Unknown scoreset type: {scoreset_type}; must be 'BasicScoreset' or 'PillarProject'")
    fits = []
    bootstrap = kwargs.get("bootstrap", True)
    for fitNum in tqdm(range(kwargs.get("num_iterations", 10)), desc="Fit iterations"):
        fit = Fit(scoreset)
        fit.run(core_limit=core_limit, num_fits=num_fits, component_range=component_range,bootstrap=bootstrap)
        fits.append(fit)
    if kwargs.get("summarize", False):
        summarize_fits(fits, scoreset, summary_file_savepath=summary_filepath, # type: ignore
                       figure_savepath=fig_filepath)
        print(f"Summary saved to {summary_filepath}")
        print(f"Figure saved to {fig_filepath}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process a scoreset and generate summary and figure.")
    parser.add_argument("scoreset_filepath", type=Path, help="Path to the input scoreset file.")
    parser.add_argument("summary_filepath", type=Path, help="Path to save the summary file.")
    parser.add_argument("fig_filepath", type=Path, help="Path to save the generated figure.")
    parser.add_argument("--scoreset_type", type=str,
                        choices=["BasicScoreset", "PillarProject"],
                        default="BasicScoreset", help="Type of scoreset. Default is 'BasicScoreset'.")
    parser.add_argument("--num_fits", type=int, default=10, help="Number of fits to perform. Default is 10.")
    parser.add_argument("--core_limit", type=int, default=1, help="Maximum number of cores to use. Default is 1.")
    parser.add_argument("--component_range", type=int, nargs='+', default=[2, 3], help="Range of components to consider. Default is [2, 3].")
    parser.add_argument("--summarize", action="store_true", help="Run the summary and save results.")

    args = parser.parse_args()

    main(
        scoreset_filepath=args.scoreset_filepath,
        summary_filepath=args.summary_filepath,
        fig_filepath=args.fig_filepath,
        num_fits=args.num_fits,
        core_limit=args.core_limit,
        component_range=args.component_range,
        scoreset_type=args.scoreset_type,
        summarize=args.summarize
    )