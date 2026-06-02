#!/bin/bash
#SBATCH --job-name=aux_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=outputs/logs/aux_sweep_%A_%a.out
#SBATCH --error=outputs/logs/aux_sweep_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d5.de.boer@student.vu.nl

WEIGHTS=(0.0 0.1 0.3 0.5 1.0)
WEIGHT=${WEIGHTS[$SLURM_ARRAY_TASK_ID]}
TAG="auxw${WEIGHT}"

CONDA_ENV="umc_thesis"

eval "$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "=== GPU info ==="
nvidia-smi
echo "================"
echo "aux_loss_weight = $WEIGHT  (tag: $TAG)"

cd "$SLURM_SUBMIT_DIR"

python scripts/05_train_composition.py \
    --model simple \
    --aux-loss-weight "$WEIGHT" \
    --tag "$TAG"
