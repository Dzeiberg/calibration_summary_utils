#!/bin/bash
#SBATCH --job-name=calibration_summary
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-20%4

SCORESETS_DIR=/projects/talisman/dzeiberg/pillar_project_data/dataset_09042025/scoresets
RESULTS_DIR=/projects/talisman/dzeiberg/calibration_results/$(date +%Y%m%d_%H%M%S)
KWARGS_FILE=/home/d.zeiberg/calibration_summary_utils/kwargs.json
module load anaconda3/2024.06
conda activate assay_calibration

python /home/d.zeiberg/calibration_summary_utils/run.py \
    --scoresets_dir "$SCORESETS_DIR" \
    --results_dir "$RESULTS_DIR" \
    --kwargs_file "$KWARGS_FILE"
