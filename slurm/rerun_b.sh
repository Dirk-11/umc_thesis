#!/bin/bash
# Re-run the composition random-labels training (failed due to model_type bug)
# and chain B_eval_main + B_eval_rand.
#
# Also migrates all checkpoint directories on Snellius to the new nested layout
# (outputs/checkpoints/{binary,composition,multiclass}) to match the updated config.
#
# Usage: bash slurm/rerun_b.sh

set -euo pipefail

CONDA_ENV="umc_thesis"
PROJECT_DIR="$HOME/umc_thesis"
LOG_DIR="$PROJECT_DIR/outputs/logs"
CKPT_ROOT="$PROJECT_DIR/outputs/checkpoints"

mkdir -p "$LOG_DIR"

# ── Migrate checkpoint directories to new nested layout ───────────────────────
echo "Checking checkpoint layout..."

# Binary: outputs/checkpoints/{fold_0..} → outputs/checkpoints/binary/
if [ -d "$CKPT_ROOT/fold_0" ]; then
    echo "  Migrating binary checkpoints → $CKPT_ROOT/binary"
    mv "$CKPT_ROOT" "${CKPT_ROOT}_binary_temp"
    mkdir -p "$CKPT_ROOT"
    mv "${CKPT_ROOT}_binary_temp" "$CKPT_ROOT/binary"
fi

# Composition: outputs/checkpoints_composition → outputs/checkpoints/composition
OLD_COMP="$PROJECT_DIR/outputs/checkpoints_composition"
if [ -d "$OLD_COMP" ] && [ ! -d "$CKPT_ROOT/composition" ]; then
    echo "  Migrating composition checkpoints → $CKPT_ROOT/composition"
    mv "$OLD_COMP" "$CKPT_ROOT/composition"
fi

# Multiclass: outputs/checkpoints_multiclass → outputs/checkpoints/multiclass
OLD_MC="$PROJECT_DIR/outputs/checkpoints_multiclass"
if [ -d "$OLD_MC" ] && [ ! -d "$CKPT_ROOT/multiclass" ]; then
    echo "  Migrating multiclass checkpoints → $CKPT_ROOT/multiclass"
    mv "$OLD_MC" "$CKPT_ROOT/multiclass"
fi

echo "Checkpoint layout:"
ls "$CKPT_ROOT/"
echo ""

# ── SLURM helpers ─────────────────────────────────────────────────────────────
SETUP="eval \"\$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)\" \
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

# ── Submit jobs ────────────────────────────────────────────────────────────────

# Re-run only the failed piece: random-labels composition training
JID_B_RAND=$(sbatch --parsable \
    --job-name="train_comp_rand" \
    --partition=gpu_a100 \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=9 --gpus=1 \
    --mem=32G \
    --time=06:00:00 \
    --mail-type=END,FAIL \
    --mail-user=d5.de.boer@student.vu.nl \
    --output="$LOG_DIR/comp_rand_%j.out" \
    --error="$LOG_DIR/comp_rand_%j.err" \
    --wrap="$SETUP && python scripts/05_train_composition.py --model simple --random-labels")

echo "train_comp_rand: job $JID_B_RAND"

# B_eval_main: regular composition checkpoints are good — no dependency needed
JID_B_EVAL_MAIN=$(sbatch --parsable \
    --job-name="B_eval_main" \
    --partition=gpu_a100 \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=9 --gpus=1 \
    --mem=32G \
    --time=00:30:00 \
    --output="$LOG_DIR/B_eval_main_%j.out" \
    --error="$LOG_DIR/B_eval_main_%j.err" \
    --wrap="$SETUP && python scripts/06_evaluate_composition.py --model simple")

echo "B_eval_main:     job $JID_B_EVAL_MAIN"

# B_eval_rand: must wait for random training to finish
JID_B_EVAL_RAND=$(submit_eval "B_eval_rand" "$JID_B_RAND" \
    "python scripts/06_evaluate_composition.py --model simple --random-labels")

echo "B_eval_rand:     job $JID_B_EVAL_RAND"
echo ""
echo "Monitor: squeue -u \$USER"
