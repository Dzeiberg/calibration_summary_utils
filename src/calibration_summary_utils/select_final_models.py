import json
from pathlib import Path
import random
from tqdm import tqdm
from typing import Optional
import argparse


def select_final_models(
    all_models_file: Path, final_models_file: Path, num_models: int = 2000
):
    all_models_file = Path(all_models_file)
    if not all_models_file.exists():
        raise FileNotFoundError(f"The file {all_models_file} does not exist.")
    with open(all_models_file, "r") as file:
        all_models = json.load(file)
    final_models = {}
    for dataset, models in all_models.items():
        if len(models) > num_models:
            selected_models = random.sample(models, num_models)
        else:
            continue
        final_models[dataset] = selected_models
    with open(final_models_file, "w") as out_file:
        json.dump(final_models, out_file, indent=4)
    print(f"Final models written to {final_models_file}")


def aggregate_and_select_final_models(
    models_dir: str | Path,
    final_models_file: str | Path,
    num_models: Optional[int | None] = None,
):
    """Aggregate all models from a directory and select a subset of final models.

    Args:
        models_dir (str|Path): Directory containing model JSON files.
        final_models_file (str|Path): Output file for the final models.
        num_models (Optional int): Number of models to select for each dataset, all models will be selected if None.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"The directory {models_dir} does not exist.")

    all_models = {}
    for model_file in tqdm(list(models_dir.glob("*.json")), desc="Aggregating models"):
        with open(model_file, "r") as file:
            model = json.load(file)
        dataset_name = model_file.name.split("_20250")[0]
        if dataset_name not in all_models:
            all_models[dataset_name] = []
        all_models[dataset_name].append(model)
    for k, v in all_models.items():
        if num_models is not None and len(v) < num_models:
            raise ValueError(
                f"Not enough models for dataset {k}. Found {len(v)}, expected at least {num_models}."
            )
    final_models = {}
    if num_models is not None:
        final_models = {k: random.sample(v, num_models) for k, v in all_models.items()}
    with open(final_models_file, "w") as out_file:
        json.dump(
            final_models if num_models is not None else all_models, out_file, indent=4
        )


if __name__ == "__main__":

    def main():
        parser = argparse.ArgumentParser(
            description="Aggregate models and select a subset of final models."
        )
        parser.add_argument(
            "models_dir", type=str, help="Directory containing model JSON files."
        )
        parser.add_argument(
            "final_models_file", type=str, help="Output file for the final models."
        )
        parser.add_argument(
            "--num_models",
            type=int,
            default=None,
            help="Number of models to select for each dataset. Defaults to all models.",
        )

        args = parser.parse_args()
        aggregate_and_select_final_models(
            args.models_dir, args.final_models_file, args.num_models
        )

    main()
