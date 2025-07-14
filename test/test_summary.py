import sys
from pathlib import Path
from tqdm import tqdm


from assay_calibration.data_utils.dataset import (
    BasicScoreset,
)
from assay_calibration.fit_utils.fit import Fit

sys.path.append(str(Path(__file__).resolve().parents[2]))
from calibration_summary_utils.src.calibration_summary_utils.summarize_fits import (
    summarize_fits,
)


def test_pipeline():
    example_data = Path(__file__).parent / "example_table.csv"
    scoreset = BasicScoreset.from_csv(example_data)
    fits = []
    for fitNum in tqdm(range(1, 6)):
        fit = Fit(scoreset)
        fit.run(core_limit=1, num_fits=1, component_range=[2, 3])
        fits.append(fit)
    summary_filepath = Path(__file__).parent / "summary.json"
    fig_filepath = Path(__file__).parent / "fit_visualization.png"
    summarize_fits(
        fits,
        scoreset,
        summary_file_savepath=summary_filepath,
        figure_savepath=fig_filepath,
    )


if __name__ == "__main__":
    test_pipeline()
    print("Test completed successfully.")
