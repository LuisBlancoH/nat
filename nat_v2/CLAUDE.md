# NAT v2 — Nested Adaptive Transformer

## What this is
A research prototype: a frozen Qwen3-4B transformer with adaptive memory neurons 
that learn at inference time via outer product writes. No gradient descent at deployment.

## Architecture
See NAT_v2_Spec.md for the complete specification. Read it fully before writing any code.

## Key rules
- Base model: Qwen/Qwen3-4B, frozen, bf16, enable_thinking=False for training
- All state updates use = not += (autograd compatibility)
- No KV cache during training — each chunk is independent
- Batch dimension on ALL state tensors
- Hook-based architecture — do NOT manually iterate through base model layers
- Use torch.bmm for batched matrix operations

## Implementation order
1. FastNeuron class (model/fast_neuron.py)
2. SlowNeuron class (model/slow_neuron.py) 
3. NATv2Model with hooks (model/nat_model.py)
4. Phase 1 training loop (training/phase1_adaptation.py)
5. Phase 2 training loop (training/phase2_consolidation.py)

## Hardware
A100 80GB. Total memory budget ~54GB. Batch size 4, chunk size 256 tokens.

## Testing
After each component, run a quick forward pass test to verify shapes and gradients flow.
