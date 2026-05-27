#!/bin/bash
# Submit all training jobs in parallel, then chain evaluation and comparison.
# Training scripts run simultaneously on separate GPUs; eval waits for them.
# Usage: bash slurm/submit_parallel.sh

set -euo pipefail

CONDA_ENV="umc_thesis"
PROJECT_DIR="$HOME/umc_thesis"
LOG_DIR="$PROJECT_DIR/outputs/logs"

mkdir -p "$LOG_DIR"

# Preamble injected into every --wrap eval job
SETUP="module load 2023 && module load Anaconda3 \
  && source \$(conda info --base)/etc/profile.d/conda.sh \
  && conda activate $CONDA_ENV \
  && cd $PROJECT_DIR"

submit_eval() {
    local name=$1 dep=$2 cmd=$3
    sbatch --parsable \
        --job-name="$name" \
        --partition=gpu_a100 \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=9 --gpus=1 \
        --mem=32G \
        --time=00:30:00 \
        --output="$LOG_DIR/${name}_%j.out" \
        --error="$LOG_DIR/${name}_%j.err" \
        --dependency="afterok:$dep" \
        --wrap="$SETUP && $cmd"
}

# ── Submit training jobs in parallel ─────────────────────────────────────────
echo "Submitting training jobs in parallel..."

JID_A=$(sbatch --parsable slurm/train_binary.sh)
JID_B=$(sbatch --parsable slurm/train_composition.sh)
JID_C=$(sbatch --parsable slurm/train_multiclass.sh)

echo "  Model A (binary):      job $JID_A"
echo "  Model B (composition): job $JID_B"
echo "  Model C (multiclass):  job $JID_C"

# ── Submit evaluation jobs (depend on their training job) ─────────────────────
echo "Submitting evaluation jobs..."

JID_A_EVAL_MAIN=$(submit_eval "A_eval_main" "$JID_A" \
    "python scripts/06_evaluate_binary.py")
JID_A_EVAL_T90=$(submit_eval  "A_eval_t90"  "$JID_A" \
    "python scripts/06_evaluate_binary.py --purity-threshold 90")
JID_A_EVAL_RAND=$(submit_eval "A_eval_rand" "$JID_A" \
    "python scripts/06_evaluate_binary.py --random-labels")

JID_B_EVAL_MAIN=$(submit_eval "B_eval_main" "$JID_B" \
    "python scripts/06_evaluate_composition.py --model simple")
JID_B_EVAL_RAND=$(submit_eval "B_eval_rand" "$JID_B" \
    "python scripts/06_evaluate_composition.py --model simple --random-labels")

JID_C_EVAL_MAIN=$(submit_eval "C_eval_main" "$JID_C" \
    "python scripts/06_evaluate_multiclass.py")
JID_C_EVAL_RAND=$(submit_eval "C_eval_rand" "$JID_C" \
    "python scripts/06_evaluate_multiclass.py --random-labels")

echo "  A eval jobs: $JID_A_EVAL_MAIN, $JID_A_EVAL_T90, $JID_A_EVAL_RAND"
echo "  B eval jobs: $JID_B_EVAL_MAIN, $JID_B_EVAL_RAND"
echo "  C eval jobs: $JID_C_EVAL_MAIN, $JID_C_EVAL_RAND"

# ── ROC comparison plot (after all Model A evals) ────────────────────────────
ALL_A="${JID_A_EVAL_MAIN}:${JID_A_EVAL_T90}:${JID_A_EVAL_RAND}"
sbatch \
    --job-name="roc_compare" \
    --partition=gpu_a100 \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=2 \
    --mem=4G \
    --time=00:10:00 \
    --output="$LOG_DIR/roc_compare_%j.out" \
    --dependency="afterok:$ALL_A" \
    --wrap="$SETUP && python scripts/07_compare_roc.py \
        --runs '100% purity:' '90% purity:_t90' 'Random:_random'"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
