import argparse
from main import main
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Process scoresets and generate summaries and figures.")
    parser.add_argument("--scoresets_dir", type=str, required=True, help="Path to the scoresets directory.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory.")
    parser.add_argument("--scoresets", type=str, nargs='+', required=False, help="List of scoresets to process.")
    parser.add_argument("--job_index", type=int, required=False, help="Index of the first scoreset to process. If not provided, start at the beginning.")
    parser.add_argument("--kwargs_file", type=str, required=True, help="JSON string of additional arguments for processing.")
    parser.add_argument("--summarize", action="store_true", help="Run the summary and save results.")
    args = parser.parse_args()
    kwargs_filepath = Path(args.kwargs_file).expanduser()
    if not kwargs_filepath.exists():
        raise ValueError(f"Kwargs file {kwargs_filepath} does not exist.")
    
    with open(kwargs_filepath, 'r') as f:
        args.kwargs = f.read()
    return args

def main_script(**kwargs):
    if len(kwargs):
        args = argparse.Namespace(**kwargs)
    else:
        args = parse_args()
    print("Running with provided kwargs:")
    print(args)
    
    scoresets_dir = Path(args.scoresets_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(exist_ok=True, parents=True)
    scoresets = args.scoresets
    if scoresets is None:
        scoresets = [file.stem for file in scoresets_dir.glob("*.json")]
    scoresets = sorted(scoresets)
    job_index = args.job_index
    job_index = 0 if job_index is None else job_index % len(scoresets)
    scoresets = scoresets[args.job_index:] + scoresets[:args.job_index]
    kwargs = json.loads(args.kwargs)
    if not scoresets_dir.exists():
        raise ValueError(f"Scoresets directory {scoresets_dir} does not exist; please download the Pillar Project data to this location.")

    for scoreset in scoresets:
        print(f"Processing {scoreset}...")
        scoreset_filepath = scoresets_dir / f"{scoreset}.json"
        summary_filepath = results_dir / f"{scoreset}_summary.json"
        fig_filepath = results_dir / f"{scoreset}_figure.png"
        fits_savedir=results_dir/"fits"
        fits_savedir.mkdir(parents=True,exist_ok=True)
        main(scoreset_filepath, fits_save_dir=fits_savedir,summary_filepath=summary_filepath, fig_filepath=fig_filepath, **kwargs)
        print(f"Finished processing {scoreset}.")
        if args.summarize:
            if not summary_filepath.exists():
                raise ValueError(f"Summary file {summary_filepath} was not created.")
            if not fig_filepath.exists():
                raise ValueError(f"Figure file {fig_filepath} was not created.")
            print(f"Summary saved to {summary_filepath}")
            print(f"Figure saved to {fig_filepath}")

if __name__ == "__main__":
    main_script()

