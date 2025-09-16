import argparse
from main import main
from pathlib import Path
import json
import time
from src.index_to_scoreset import get_scoreset_name

def parse_args():
    parser = argparse.ArgumentParser(description="Process scoresets and generate summaries and figures.")
    parser.add_argument("--scoresets_dir", type=str, required=True, help="Path to the scoresets directory.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory.")
    parser.add_argument("--job_index", type=int, required=True, help="Index of this job in the job array")
    parser.add_argument("--runs_per_job", type=int, default=1, help="Number of scoresets to process per job. Default is 1.")
    parser.add_argument("--jobs_filepath", type=str, required=True, help="Path to the runs_needed.tsv file.")
    parser.add_argument("--kwargs_file", type=str, required=True, help="JSON string of additional arguments for processing.")
    parser.add_argument("--summarize", action="store_true", help="Run the summary and save results.")
    args = parser.parse_args()
    kwargs_filepath = Path(args.kwargs_file).expanduser()
    if not kwargs_filepath.exists():
        raise ValueError(f"Kwargs file {kwargs_filepath} does not exist.")
    
    with open(kwargs_filepath, 'r') as f:
        args.kwargs = f.read()
    return args

def get_scoresets_list(runs_filepath: Path, job_index: int, runs_per_job: int) -> list[str]:
    runs_filepath = Path(runs_filepath).expanduser()
    if not runs_filepath.exists():
        raise ValueError(f"Runs needed TSV file {runs_filepath} does not exist.")
    scoresets = []
    for idx in range(job_index * runs_per_job, job_index * runs_per_job + runs_per_job):
        scoreset_name = get_scoreset_name(runs_filepath, idx)
        scoresets.append(scoreset_name)
    return scoresets

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
    scoresets = get_scoresets_list(args.jobs_filepath, args.job_index, args.runs_per_job) 
    print(f"Scoresets to process in this job: {scoresets}")
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
        start_time = time.time()
        main(scoreset_filepath, fits_save_dir=fits_savedir,summary_filepath=summary_filepath, fig_filepath=fig_filepath,**kwargs)
        print(f"Finished processing {scoreset}.")
        print(f"Time taken: {time.time() - start_time:.2f} seconds.")
        if args.summarize:
            if not summary_filepath.exists():
                raise ValueError(f"Summary file {summary_filepath} was not created.")
            if not fig_filepath.exists():
                raise ValueError(f"Figure file {fig_filepath} was not created.")
            print(f"Summary saved to {summary_filepath}")
            print(f"Figure saved to {fig_filepath}")

if __name__ == "__main__":
    main_script()

