from pillar_project.pillar_project.data_utils.dataset import Scoreset
from pathlib import Path
import pandas as pd

def generate_scoreset_files(pillar_df_filepath, scoreset_save_dir, scoreset_summary_dir=None):
    """
    Generate scoreset json files for all scoresets in the given dataframe.
    Args
        pillar_df_filepath (str): Path to the pillar dataframe file.
        scoreset_save_dir (str): Directory to save the scoreset json files.
        scoreset_summary_dir (Optional[str]): Directory to save the scoreset summaries, including counts for each class.
            If None, defaults to "/tmp/summaries".
    """
    pillar_df_filepath = Path(pillar_df_filepath)
    scoreset_save_dir = Path(scoreset_save_dir)
    if scoreset_summary_dir is not None:
        scoreset_summary_dir = Path(scoreset_summary_dir)
    else:
        scoreset_summary_dir = Path("/tmp/summaries")
    scoreset_summary_dir.mkdir(parents=True, exist_ok=True)
    if not pillar_df_filepath.exists():
        raise FileNotFoundError(f"File {pillar_df_filepath} not found")
    
    pillar_df = pd.read_csv(pillar_df_filepath)
    scoreset_save_dir.mkdir(parents=True, exist_ok=True)
    for scoreset_id, group in pillar_df.groupby("Dataset"):
        scoreset = Scoreset(group)
        scoreset_filepath = scoreset_save_dir / f"{scoreset_id}.json"
        scoreset.to_json(scoreset_filepath)
        print(f"Saved scoreset {scoreset_id} to {scoreset_filepath}")
        with open(scoreset_summary_dir / f"{scoreset_id}_summary.json", "w") as f:
            f.write(str(scoreset))
        print(f"Saved scoreset summary {scoreset_id} to {scoreset_summary_dir / f'{scoreset_id}_summary.json'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate scoreset JSON files and summaries.")
    parser.add_argument("pillar_df_filepath", type=str, help="Path to the pillar dataframe file.")
    parser.add_argument("scoreset_save_dir", type=str, help="Directory to save the scoreset JSON files.")
    parser.add_argument("--scoreset_summary_dir", type=str, default=None, 
                        help="Directory to save the scoreset summaries. Defaults to '/tmp/summaries' if not provided.")

    args = parser.parse_args()

    generate_scoreset_files(args.pillar_df_filepath, args.scoreset_save_dir, args.scoreset_summary_dir)

    # Example usage:
        # python generate_scoreset_files.py \
            # data/pillar_data_clinvar38_053125_wREVEL_gold_standards.csv" \
            # data/scoresets \
            # data/scoreset_summaries
    # This will generate scoreset json files and summaries in the specified directories.