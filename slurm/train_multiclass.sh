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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d5.de.boer@student.vu.nl

CONDA_ENV="umc_thesis"

eval "$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "=== GPU info ==="
nvidia-smi
echo "================"

cd "$SLURM_SUBMIT_DIR"

python scripts/05_train_multiclass.py
python scripts/05_train_multiclass.py --random-labels
