from pathlib import Path
from datetime import datetime
import json
from src.calibration_summary_utils.summarize_fits import summarize_pillar_fits
import argparse

def get_models(results_dir, scoreset_name):
    results_dir = Path(results_dir).expanduser()
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist.")
    model_files = list(results_dir.glob(f"{scoreset_name}_*_summary.json"))
    models = []
    for model_file in model_files:
        with open(model_file, 'r') as f:
            model_data = json.load(f)
            models.append(model_data)
    return models

def summarize_models(scoresets_dir : str|Path, models_dir : str|Path,
                     results_savedir : str|Path, dataframe_filepath : str|Path,**kwargs):
    """

    Summarize models for a given scoreset

    Args:
        scoresets_dir (str|Path): Directory containing the scoresets
        models_dir (str|Path): Directory containing the model fits
        results_savedir (str|Path): Directory to save the summary results
        dataframe_filepath (str|Path): Path to the Pillar Project dataframe
        **kwargs: Additional keyword arguments to pass to summarize_pillar_fits
    """
    
    scoresets_dir = Path(scoresets_dir).expanduser()
    models_dir = Path(models_dir).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_savedir = Path(results_savedir).expanduser()
    results_savedir = results_savedir / f"results_{timestamp}"
    summaries_savedir = results_savedir / "summaries"
    figures_savedir = results_savedir / "figures"
    figures_savedir.mkdir(parents=True, exist_ok=True)
    summaries_savedir.mkdir(parents=True, exist_ok=True)
    if not scoresets_dir.exists():
        raise ValueError(f"Scoresets directory {scoresets_dir} does not exist.")
    if not models_dir.exists():
        raise ValueError(f"Models directory {models_dir} does not exist.")
    dataframe_filepath = Path(dataframe_filepath).expanduser()
    if not dataframe_filepath.exists():
        raise ValueError(f"Dataframe file {dataframe_filepath} does not exist.")
    final_models_savefile = results_savedir / f"final_models_{timestamp}.json"
    with open(results_savedir / "config.txt", "w") as f:
        json.dump(dict(scoresets_dir=str(scoresets_dir),
                       models_dir=str(models_dir),
                       results_savedir=str(results_savedir),
                       dataframe_filepath=str(dataframe_filepath),
                       timestamp = timestamp,
                       **kwargs), f, indent=4)
    summarize_pillar_fits(models_dir,scoresets_dir,final_models_savefile,
                          summaries_savedir,figures_savedir,dataframe_filepath, **kwargs)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Summarize models for a given scoreset.")
    parser.add_argument("scoresets_dir", type=str, help="Directory containing the scoresets")
    parser.add_argument("models_dir", type=str, help="Directory containing the model fits")
    parser.add_argument("results_savedir", type=str, help="Directory to save the summary results")
    parser.add_argument("dataframe_filepath", type=str, help="Path to the Pillar Project dataframe")
    parser.add_argument("--additional_args", nargs="*", help="Additional keyword arguments for summarize_pillar_fits")
    args = parser.parse_args()

    additional_kwargs = {}
    if args.additional_args:
        for arg in args.additional_args:
            key, value = arg.split("=")
            additional_kwargs[key] = value

    summarize_models(
        scoresets_dir=args.scoresets_dir,
        models_dir=args.models_dir,
        results_savedir=args.results_savedir,
        dataframe_filepath=args.dataframe_filepath,
        **additional_kwargs
    )