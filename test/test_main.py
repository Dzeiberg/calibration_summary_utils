from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from main import main

def test_main():
    pwd = Path(__file__).parent
    scoreset_filepath = pwd / "BRCA1_Findlay_2018.json"
    summary_filepath = pwd / "test_results/summary.json"
    fig_filepath = pwd / "test_results/figure.png"
    if not scoreset_filepath.exists():
        print(f"Test skipped: {scoreset_filepath} does not exist.")
        return
    main(scoreset_filepath,summary_filepath,fig_filepath,
         num_fits=2,core_limit=1,component_range=[2,3],scoreset_type="PillarProject",bootstrap=False)
    assert summary_filepath.exists(), "Summary file was not created."
    assert fig_filepath.exists(), "Figure file was not created."
    print("Test passed: Summary and figure files were created.")

if __name__ == "__main__":
    test_main()