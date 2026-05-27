#!/bin/bash
#SBATCH --job-name=train_all
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/logs/all_models_%j.out
#SBATCH --error=outputs/logs/all_models_%j.err

CONDA_ENV="kidney_stone"   # <-- update if your env has a different name

module load 2023
module load Anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "=== GPU info ==="
nvidia-smi
echo "================"

cd "$SLURM_SUBMIT_DIR"

# ── Model A — Binary ─────────────────────────────────────────────────────────
echo ">>> Model A: binary (100% purity)"
python scripts/05_train_binary.py

echo ">>> Model A: binary (90% purity)"
python scripts/05_train_binary.py --purity-threshold 90

echo ">>> Model A: binary (random labels baseline)"
python scripts/05_train_binary.py --random-labels

# ── Model B — Composition regression ─────────────────────────────────────────
echo ">>> Model B: composition regression"
python scripts/05_train_composition.py --model simple

echo ">>> Model B: composition regression (random labels baseline)"
python scripts/05_train_composition.py --model simple --random-labels

# ── Model C — Multiclass ─────────────────────────────────────────────────────
echo ">>> Model C: multiclass"
python scripts/05_train_multiclass.py

echo ">>> Model C: multiclass (random labels baseline)"
python scripts/05_train_multiclass.py --random-labels

echo "=== All training complete ==="
