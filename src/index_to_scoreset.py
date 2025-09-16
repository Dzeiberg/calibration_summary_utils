import pandas as pd
from pathlib import Path
def get_scoreset_name(runs_needed_tsv: Path, job_index : int) -> str:
    df = pd.read_csv(runs_needed_tsv, sep="\t")
    df.columns = ['scoreset_name', 'runs_needed']
    df.runs_needed = df.runs_needed - 1 # make zero-indexed
    cumulative_runs = df['runs_needed'].cumsum()
    scoreset_row = df[cumulative_runs >= job_index].iloc[0]
    return scoreset_row['scoreset_name']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get scoreset name for a given job index.")
    parser.add_argument("runs_needed_tsv", type=str, help="Path to the runs_needed.tsv file.")
    parser.add_argument("job_index", type=int, help="Job index to look up.")
    args = parser.parse_args()
    runs_needed_tsv = Path(args.runs_needed_tsv).expanduser()
    if not runs_needed_tsv.exists():
        raise ValueError(f"Runs needed TSV file {runs_needed_tsv} does not exist.")
    scoreset_name = get_scoreset_name(runs_needed_tsv, args.job_index)
    print(scoreset_name)