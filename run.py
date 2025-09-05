import argparse
from main import main
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Process scoresets and generate summaries and figures.")
    parser.add_argument("--scoresets_dir", type=str, required=True, help="Path to the scoresets directory.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory.")
    parser.add_argument("--scoresets", type=str, nargs='+', required=False, help="List of scoresets to process.")
    parser.add_argument("--kwargs_file", type=str, required=True, help="JSON string of additional arguments for processing.")
    args = parser.parse_args()
    kwargs_filepath = Path(args.kwargs_file).expanduser()
    if not kwargs_filepath.exists():
        raise ValueError(f"Kwargs file {kwargs_filepath} does not exist.")
    
    with open(kwargs_filepath, 'r') as f:
        args.kwargs = f.read()
    return args

def main_script():
    args = parse_args()
    
    scoresets_dir = Path(args.scoresets_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    scoresets = args.scoresets
    if scoresets is None:
        scoresets = [file.stem for file in scoresets_dir.glob("*.json")]
    kwargs = json.loads(args.kwargs)

    results_dir.mkdir(exist_ok=True, parents=True)
    if not scoresets_dir.exists():
        raise ValueError(f"Scoresets directory {scoresets_dir} does not exist; please download the Pillar Project data to this location.")

    for scoreset in scoresets:
        print(f"Processing {scoreset}...")
        scoreset_filepath = scoresets_dir / f"{scoreset}.json"
        summary_filepath = results_dir / f"{scoreset}_summary.json"
        fig_filepath = results_dir / f"{scoreset}_figure.png"
        main(scoreset_filepath, summary_filepath, fig_filepath, **kwargs)
        print(f"Summary saved to {summary_filepath}")
        print(f"Figure saved to {fig_filepath}")

if __name__ == "__main__":
    main_script()

