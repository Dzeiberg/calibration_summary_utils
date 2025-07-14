import json
from pathlib import Path
import numpy as np
from tqdm import tqdm


def summarize_priors(summaries_dir: Path, write_filepath: Path | None = None):
    """
    Summarizes the prior distributions from JSON files in the specified directory.

    Args:
        summaries_dir (Path): Path to the directory containing JSON files with prior distributions.

    Returns:
        dict: A dictionary containing the mean and standard deviation of each prior distribution.
    """
    prior_summaries = {}
    for file_path in tqdm(list(summaries_dir.glob("*.json"))):
        with open(file_path, "r") as file:
            data = json.load(file)
            priors = data.get("priors", [])
            prior_summaries[file_path.stem] = {
                "mean": np.mean(priors),
                "median": np.median(priors),
                "std": np.std(priors),
            }
    # Write the summaries to a JSON file
    if write_filepath is not None:
        with open(write_filepath, "w") as outfile:
            json.dump(prior_summaries, outfile, indent=4)
        print(f"Prior summaries written to {write_filepath}")
    return prior_summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize prior distributions from JSON files."
    )
    parser.add_argument(
        "summaries_dir",
        type=Path,
        help="Directory containing JSON files with prior distributions.",
    )
    parser.add_argument(
        "--write_filepath",
        type=Path,
        default=None,
        help="Filepath to write the summary JSON file.",
    )

    args = parser.parse_args()

    summarize_priors(args.summaries_dir, args.write_filepath)
