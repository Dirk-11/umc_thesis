#!/bin/bash
#SBATCH --job-name=kidney_all_models
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/home/ddboer/umc_thesis/outputs/logs/all_models_%j.out
#SBATCH --error=/home/ddboer/umc_thesis/outputs/logs/all_models_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load 2023
module load Anaconda3/2023.07-2
conda activate umc_thesis

# ── Go to project root ────────────────────────────────────────────────────────
cd /home/ddboer/umc_thesis

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# ── Run all models sequentially ───────────────────────────────────────────────

echo "========================================"
echo "Starting Model A (binary) at $(date)"
echo "========================================"
python scripts/05_train_binary.py
echo "Finished Model A at $(date)"

echo "========================================"
echo "Starting Model B (composition) at $(date)"
echo "========================================"
python scripts/05_train_composition.py
echo "Finished Model B at $(date)"

echo "========================================"
echo "Starting Model C (multiclass) at $(date)"
echo "========================================"
python scripts/05_train_multiclass.py
echo "Finished Model C at $(date)"

echo "========================================"
echo "All models complete at $(date)"
echo "========================================"
