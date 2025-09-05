#!/bin/bash

TOTAL_JOBS=100
MAX_CONCURRENT_JOBS=50

SCORESETS_DIR=/projects/talisman/dzeiberg/pillar_project_data/dataset_09042025/scoresets
RESULTS_DIR_BASE=/projects/talisman/dzeiberg/calibration_results/test/
KWARGS_FILE=/home/d.zeiberg/calibration_summary_utils/test_kwargs.json
module load anaconda3/2024.06
conda activate assay_calibration

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$RESULTS_DIR_BASE/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# Split jobs into batches of MAX_CONCURRENT_JOBS
for ((i=0; i<TOTAL_JOBS; i+=MAX_CONCURRENT_JOBS)); do
    for ((j=0; j<MAX_CONCURRENT_JOBS && i+j<TOTAL_JOBS; j++)); do
        ITERATION=$((i + j))
        sbatch --export=ALL,SCORESETS_DIR="$SCORESETS_DIR",RESULTS_DIR="$RESULTS_DIR",KWARGS_FILE="$KWARGS_FILE",ITERATION="$ITERATION" <<'EOF'
#!/bin/bash
#SBATCH --job-name=calibration_summary
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

python /home/d.zeiberg/calibration_summary_utils/run.py \
    --scoresets_dir "$SCORESETS_DIR" \
    --results_dir "$RESULTS_DIR/iteration_$ITERATION" \
    --kwargs_file "$KWARGS_FILE"
EOF
    done
    # Wait for the current batch of jobs to finish before submitting the next batch
    while [ "$(squeue -u $USER | grep -c calibration_summary)" -ge "$MAX_CONCURRENT_JOBS" ]; do
        sleep 10
    done
done
