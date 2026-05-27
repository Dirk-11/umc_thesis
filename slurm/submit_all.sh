#!/bin/bash
# Submit all training + evaluation jobs for the kidney stone thesis.
# Training jobs run in parallel; evaluation jobs depend on their training job.
# Usage: bash slurm/submit_all.sh

set -euo pipefail

# ── Adjust these to match your Snellius environment ──────────────────────────
PARTITION="gpu"
TIME_TRAIN="02:00:00"
TIME_EVAL="00:30:00"
CPUS=9
PROJECT_DIR="$HOME/umc_thesis"   # absolute path on Snellius
LOG_DIR="$PROJECT_DIR/outputs/logs"
# If you use a venv, set this; otherwise use the module system
PYTHON="python"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

submit_train() {
    local name=$1; shift
    sbatch --parsable \
        --job-name="$name" \
        --partition="$PARTITION" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="$CPUS" \
        --gpus=1 \
        --time="$TIME_TRAIN" \
        --output="$LOG_DIR/slurm_${name}_%j.out" \
        --error="$LOG_DIR/slurm_${name}_%j.err" \
        --wrap="cd $PROJECT_DIR && $PYTHON scripts/$*"
}

submit_eval() {
    local name=$1; local dep=$2; shift 2
    sbatch --parsable \
        --job-name="$name" \
        --partition="$PARTITION" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task="$CPUS" \
        --gpus=1 \
        --time="$TIME_EVAL" \
        --output="$LOG_DIR/slurm_${name}_%j.out" \
        --error="$LOG_DIR/slurm_${name}_%j.err" \
        --dependency="afterok:$dep" \
        --wrap="cd $PROJECT_DIR && $PYTHON scripts/$*"
}

echo "Submitting training jobs..."

# ── Model A — Binary ─────────────────────────────────────────────────────────
JID_A_MAIN=$(submit_train    "A_train_main"   "05_train_binary.py")
JID_A_T90=$(submit_train     "A_train_t90"    "05_train_binary.py --purity-threshold 90")
JID_A_RAND=$(submit_train    "A_train_rand"   "05_train_binary.py --random-labels")

# ── Model B — Composition regression ─────────────────────────────────────────
JID_B_MAIN=$(submit_train    "B_train_main"   "05_train_composition.py --model simple")
JID_B_RAND=$(submit_train    "B_train_rand"   "05_train_composition.py --model simple --random-labels")

# ── Model C — Multiclass ──────────────────────────────────────────────────────
JID_C_MAIN=$(submit_train    "C_train_main"   "05_train_multiclass.py")
JID_C_RAND=$(submit_train    "C_train_rand"   "05_train_multiclass.py --random-labels")

echo "Training jobs: A=$JID_A_MAIN,$JID_A_T90,$JID_A_RAND  B=$JID_B_MAIN,$JID_B_RAND  C=$JID_C_MAIN,$JID_C_RAND"
echo ""
echo "Submitting evaluation jobs (depend on training)..."

# ── Model A — Evaluation ─────────────────────────────────────────────────────
JID_A_EVAL_MAIN=$(submit_eval "A_eval_main"   "$JID_A_MAIN"  "06_evaluate_binary.py")
JID_A_EVAL_T90=$(submit_eval  "A_eval_t90"    "$JID_A_T90"   "06_evaluate_binary.py --purity-threshold 90")
JID_A_EVAL_RAND=$(submit_eval "A_eval_rand"   "$JID_A_RAND"  "06_evaluate_binary.py --random-labels")

# ── Model B — Evaluation ─────────────────────────────────────────────────────
JID_B_EVAL_MAIN=$(submit_eval "B_eval_main"   "$JID_B_MAIN"  "06_evaluate_composition.py --model simple")
JID_B_EVAL_RAND=$(submit_eval "B_eval_rand"   "$JID_B_RAND"  "06_evaluate_composition.py --model simple --random-labels")

# ── Model C — Evaluation ─────────────────────────────────────────────────────
JID_C_EVAL_MAIN=$(submit_eval "C_eval_main"   "$JID_C_MAIN"  "06_evaluate_multiclass.py")
JID_C_EVAL_RAND=$(submit_eval "C_eval_rand"   "$JID_C_RAND"  "06_evaluate_multiclass.py --random-labels")

# ── ROC comparison plot (depends on all three A eval jobs) ───────────────────
ALL_A_EVALS="${JID_A_EVAL_MAIN}:${JID_A_EVAL_T90}:${JID_A_EVAL_RAND}"
sbatch \
    --job-name="roc_compare" \
    --partition="$PARTITION" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=2 \
    --time="00:10:00" \
    --output="$LOG_DIR/slurm_roc_compare_%j.out" \
    --dependency="afterok:$ALL_A_EVALS" \
    --wrap="cd $PROJECT_DIR && $PYTHON scripts/07_compare_roc.py \
        --runs '100% purity:' '90% purity:_t90' 'Random:_random'"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
