#!/bin/bash
#SBATCH --job-name=calibration_summary
#SBATCH --output=/projects/talisman/dzeiberg/logs/slurm-%j.out
#SBATCH --error=/projects/talisman/dzeiberg/logs/slurm-%j.err
#SBATCH --time=47:59:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-1000%4

cwd=$(dirname "$(realpath "$0")")
KWARGS_FILE=$cwd/run_kwargs.json
########### Explorer #####################
# SCORESETS_DIR=/projects/talisman/dzeiberg/pillar_project_data/dataset_09042025/scoresets
# RESULTS_DIR=/projects/talisman/dzeiberg/calibration_results/$(date +%Y%m%d_%H%M%S)
# module load anaconda3/2024.06
# conda init
# source /home/d.zeiberg/.bashrc
# conda activate assay_calibration
############ Big Ticket ################
SCORESETS_DIR=/data/dzeiberg/pillar_project/pillar_project_data/dataset_09042025/scoresets
RESULTS_DIR=/data/dzeiberg/pillar_project/pillar_project_data/results/$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR


uv run python $cwd/run.py \
    --scoresets_dir "$SCORESETS_DIR" \
    --results_dir "$RESULTS_DIR" \
    --kwargs_file "$KWARGS_FILE"
