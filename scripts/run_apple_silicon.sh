#!/usr/bin/env bash
# ============================================================================
# run_apple_silicon.sh — Full NAT training pipeline on Apple Silicon (MPS)
# ============================================================================
#
# Usage:
#   chmod +x scripts/run_apple_silicon.sh
#   ./scripts/run_apple_silicon.sh              # all phases
#   ./scripts/run_apple_silicon.sh --phase 1    # phase 1 only
#   ./scripts/run_apple_silicon.sh --phase 2    # phase 2 only (needs phase 1 checkpoint)
#   ./scripts/run_apple_silicon.sh --phase 3    # phase 3 only (needs phase 2 checkpoint)
#   ./scripts/run_apple_silicon.sh --eval       # evaluation only (needs phase 3 checkpoint)
#   ./scripts/run_apple_silicon.sh --synthetic  # all phases with synthetic data (fast test)
#
# Expected timeline (M1/M2 16 GB):
#   Phase 1: ~3-5 days
#   Phase 2: ~3-5 days
#   Phase 3: ~1-2 weeks
#   Eval:    ~1-2 hours
#   Total:   ~3-4 weeks
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$PROJECT_DIR/configs/apple_silicon.yaml"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

# Defaults
PHASE="all"
SYNTHETIC=""
WANDB=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --eval)
            PHASE="eval"
            shift
            ;;
        --synthetic)
            SYNTHETIC="--synthetic"
            shift
            ;;
        --wandb)
            WANDB="--wandb"
            shift
            ;;
        -h|--help)
            head -20 "$0" | tail -15
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo " NAT Training — Apple Silicon (MPS)"
echo "============================================"
echo " Config:    $CONFIG"
echo " Phase:     $PHASE"
echo " Synthetic: ${SYNTHETIC:-no}"
echo " W&B:       ${WANDB:-no}"
echo "============================================"

# Verify MPS is available
python3 -c "
import torch
assert torch.backends.mps.is_available(), (
    'MPS not available. Requires macOS 12.3+ and PyTorch 2.0+.'
)
print(f'MPS available — PyTorch {torch.__version__}')
"

mkdir -p "$CHECKPOINT_DIR"

# ---- Phase 1: Meta-learn the learning rule θ ----
run_phase1() {
    echo ""
    echo ">>> Phase 1: Meta-learning θ"
    echo "    Estimated time: 3-5 days on M1 16 GB"
    echo ""
    python3 "$PROJECT_DIR/scripts/train_phase1.py" \
        --config "$CONFIG" \
        $SYNTHETIC $WANDB
}

# ---- Phase 2: Episodic multi-task training ----
run_phase2() {
    local ckpt="$CHECKPOINT_DIR/phase1.pt"
    if [[ ! -f "$ckpt" ]] && [[ -z "$SYNTHETIC" ]]; then
        echo "ERROR: Phase 1 checkpoint not found at $ckpt"
        echo "       Run Phase 1 first, or use --synthetic for testing."
        exit 1
    fi
    echo ""
    echo ">>> Phase 2: Episodic training"
    echo "    Estimated time: 3-5 days on M1 16 GB"
    echo ""
    local resume_flag=""
    [[ -f "$ckpt" ]] && resume_flag="--resume $ckpt"
    python3 "$PROJECT_DIR/scripts/train_phase2.py" \
        --config "$CONFIG" \
        $resume_flag $SYNTHETIC $WANDB
}

# ---- Phase 3: Consolidation dynamics ----
run_phase3() {
    local ckpt="$CHECKPOINT_DIR/phase2.pt"
    if [[ ! -f "$ckpt" ]] && [[ -z "$SYNTHETIC" ]]; then
        echo "ERROR: Phase 2 checkpoint not found at $ckpt"
        echo "       Run Phase 2 first, or use --synthetic for testing."
        exit 1
    fi
    echo ""
    echo ">>> Phase 3: Consolidation dynamics"
    echo "    Estimated time: 1-2 weeks on M1 16 GB"
    echo ""
    local resume_flag=""
    [[ -f "$ckpt" ]] && resume_flag="--resume $ckpt"
    python3 "$PROJECT_DIR/scripts/train_phase3.py" \
        --config "$CONFIG" \
        $resume_flag $SYNTHETIC $WANDB
}

# ---- Evaluation ----
run_eval() {
    local ckpt="$CHECKPOINT_DIR/phase3.pt"
    echo ""
    echo ">>> Evaluation Suite"
    echo ""
    local ckpt_flag=""
    [[ -f "$ckpt" ]] && ckpt_flag="--checkpoint $ckpt"
    python3 "$PROJECT_DIR/scripts/evaluate.py" \
        --config "$CONFIG" \
        --output-dir "$PROJECT_DIR/results" \
        $ckpt_flag $SYNTHETIC $WANDB
}

# ---- Execute ----
case "$PHASE" in
    1)     run_phase1 ;;
    2)     run_phase2 ;;
    3)     run_phase3 ;;
    eval)  run_eval ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_eval
        echo ""
        echo "============================================"
        echo " All phases complete!"
        echo " Results: $PROJECT_DIR/results/"
        echo " Checkpoints: $CHECKPOINT_DIR/"
        echo "============================================"
        ;;
    *)
        echo "Unknown phase: $PHASE (use 1, 2, 3, eval, or all)"
        exit 1
        ;;
esac
