#!/bin/bash
#SBATCH --job-name=train_multiclass
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/multiclass_%j.out
#SBATCH --error=outputs/logs/multiclass_%j.err

CONDA_ENV="kidney_stone"   # <-- update if your env has a different name

module load 2023
module load Anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "=== GPU info ==="
nvidia-smi
echo "================"

cd "$SLURM_SUBMIT_DIR"

python scripts/05_train_multiclass.py
python scripts/05_train_multiclass.py --random-labels
