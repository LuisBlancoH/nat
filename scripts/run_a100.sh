#!/usr/bin/env bash
# ============================================================================
# run_a100.sh — Full NAT training pipeline on NVIDIA A100
# ============================================================================
#
# Usage:
#   chmod +x scripts/run_a100.sh
#   ./scripts/run_a100.sh                       # all phases
#   ./scripts/run_a100.sh --phase 1             # phase 1 only
#   ./scripts/run_a100.sh --phase 2             # phase 2 (needs phase 1 checkpoint)
#   ./scripts/run_a100.sh --phase 3             # phase 3 (needs phase 2 checkpoint)
#   ./scripts/run_a100.sh --eval                # evaluation only
#   ./scripts/run_a100.sh --synthetic           # all phases with synthetic data
#   ./scripts/run_a100.sh --gpu 1               # use a specific GPU
#
# Expected timeline (single A100 80 GB):
#   Phase 1: ~12-18 hours
#   Phase 2: ~10-14 hours
#   Phase 3: ~1-2 days
#   Eval:    ~30 minutes
#   Total:   ~2-4 days
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$PROJECT_DIR/configs/a100.yaml"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

# Defaults
PHASE="all"
SYNTHETIC=""
WANDB=""
GPU_ID=0

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
        --gpu)
            GPU_ID="$2"
            shift 2
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

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "============================================"
echo " NAT Training — NVIDIA A100 (CUDA)"
echo "============================================"
echo " Config:    $CONFIG"
echo " Phase:     $PHASE"
echo " GPU:       $GPU_ID"
echo " Synthetic: ${SYNTHETIC:-no}"
echo " W&B:       ${WANDB:-no}"
echo "============================================"

# Verify CUDA is available and print GPU info
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
gpu = torch.cuda.get_device_properties(0)
print(f'GPU: {gpu.name}')
print(f'VRAM: {gpu.total_mem / 1024**3:.1f} GB')
print(f'Compute capability: {gpu.major}.{gpu.minor}')
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
"

mkdir -p "$CHECKPOINT_DIR"

# ---- Phase 1: Meta-learn the learning rule θ ----
run_phase1() {
    echo ""
    echo ">>> Phase 1: Meta-learning θ"
    echo "    Estimated time: 12-18 hours on A100"
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
    echo "    Estimated time: 10-14 hours on A100"
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
    echo "    Estimated time: 1-2 days on A100"
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
