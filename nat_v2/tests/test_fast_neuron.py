"""
Tests for FastNeuron — verifies shapes, memory growth, gradient flow, and projection writes.

Uses small dimensions for speed. Run with: python -m tests.test_fast_neuron
"""

import torch

from model.fast_neuron import FastNeuron

# Small dims for fast CPU testing
DIMS = dict(
    d_model=64,
    rank=8,
    d_query=16,
    d_value=32,
    d_proj=16,
    d_context=16,
    d_report=16,
    d_hidden=32,
)
BATCH = 2
SEQ = 16
NUM_CHUNKS = 4


def make_neuron(**overrides):
    kw = {**DIMS, **overrides}
    return FastNeuron(**kw)


# ------------------------------------------------------------------
# Test 1: Output shapes are correct
# ------------------------------------------------------------------
def test_output_shapes():
    torch.manual_seed(42)
    neuron = make_neuron()
    neuron.start_session(BATCH, "cpu")

    h = torch.randn(BATCH, SEQ, DIMS["d_model"])
    h_new = neuron(h)

    assert h_new.shape == (BATCH, SEQ, DIMS["d_model"]), \
        f"h_new shape: {h_new.shape}"
    assert neuron.last_report.shape == (BATCH, DIMS["d_report"]), \
        f"report shape: {neuron.last_report.shape}"
    assert neuron.mem_A.shape == (BATCH, DIMS["d_model"], DIMS["rank"]), \
        f"mem_A shape: {neuron.mem_A.shape}"
    assert neuron.W_down_mod.shape == (BATCH, DIMS["d_model"], DIMS["d_proj"]), \
        f"W_down_mod shape: {neuron.W_down_mod.shape}"
    assert neuron.W_up_mod.shape == (BATCH, DIMS["d_proj"], DIMS["d_model"]), \
        f"W_up_mod shape: {neuron.W_up_mod.shape}"

    print("PASS: Test 1 — Output shapes correct")


# ------------------------------------------------------------------
# Test 2: mem_A norm increases after writes
# ------------------------------------------------------------------
def test_mem_a_norm_increases():
    torch.manual_seed(42)
    neuron = make_neuron()
    neuron.start_session(BATCH, "cpu")

    norms = []
    for _ in range(NUM_CHUNKS):
        h = torch.randn(BATCH, SEQ, DIMS["d_model"])
        neuron(h)
        norms.append(torch.norm(neuron.mem_A).item())

    assert norms[0] > 0, f"mem_A should be nonzero after first write, got {norms[0]}"
    assert norms[-1] > norms[0], \
        f"mem_A norm should grow over chunks: {norms}"

    print(f"PASS: Test 2 — mem_A norm increases: {[f'{n:.4f}' for n in norms]}")


# ------------------------------------------------------------------
# Test 3: Gradients flow back through the entire chain
# ------------------------------------------------------------------
def test_gradient_flow():
    torch.manual_seed(42)
    neuron = make_neuron()

    # Force threshold low so projection write path is active (tests more params)
    neuron.fixed_threshold = 0.0

    neuron.start_session(BATCH, "cpu")
    neuron.zero_grad()

    # Run 4 chunks — state carries the computation graph across chunks
    for _ in range(NUM_CHUNKS):
        h = torch.randn(BATCH, SEQ, DIMS["d_model"], requires_grad=True)
        h_new = neuron(h)

    # Loss on final chunk output + report (exercises all output paths)
    loss = h_new.sum() + neuron.last_report.sum()
    loss.backward()

    params_with_grad = []
    params_no_grad = []
    for name, p in neuron.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            params_with_grad.append(name)
        else:
            params_no_grad.append(name)

    # Every network in the pipeline should receive gradients, except
    # threshold_net (fixed threshold bypasses it).
    critical_prefixes = [
        "surprise_net",
        "write_key_net",
        "write_value_net",
        "lr_net",
        "read_query_net",
        "W_K",
        "W_V",
        "value_up_proj",
        "W_down_base",
        "W_up_base",
        "proj_write_down_net",
        "proj_write_up_net",
        "proj_lr_net",
        "gate_net",
        "layer_norm",
        "slot_layer_norm",
        "report_net",
    ]
    missing = []
    for prefix in critical_prefixes:
        if not any(n.startswith(prefix) for n in params_with_grad):
            missing.append(prefix)

    assert not missing, \
        f"No gradient for: {missing}. With grad: {params_with_grad}"

    total = len(params_with_grad) + len(params_no_grad)
    print(f"PASS: Test 3 — Gradients flow to {len(params_with_grad)}/{total} params")
    if params_no_grad:
        print(f"  (No grad: {params_no_grad} — expected for threshold_net hard mask)")


# ------------------------------------------------------------------
# Test 4: W_down_mod and W_up_mod modified when surprise > threshold
# ------------------------------------------------------------------
def test_projection_writes():
    # 4a: threshold ~ 0 → all writes go through
    torch.manual_seed(42)
    neuron_low = make_neuron()
    neuron_low.fixed_threshold = 0.0

    neuron_low.start_session(BATCH, "cpu")
    assert torch.norm(neuron_low.W_down_mod).item() == 0.0
    assert torch.norm(neuron_low.W_up_mod).item() == 0.0

    for _ in range(NUM_CHUNKS):
        h = torch.randn(BATCH, SEQ, DIMS["d_model"])
        neuron_low(h)

    low_down = torch.norm(neuron_low.W_down_mod).item()
    low_up = torch.norm(neuron_low.W_up_mod).item()
    assert low_down > 0, f"W_down_mod should be modified (low threshold), got {low_down}"
    assert low_up > 0, f"W_up_mod should be modified (low threshold), got {low_up}"

    # 4b: threshold ~ 1 → near-zero writes (soft threshold)
    torch.manual_seed(42)
    neuron_high = make_neuron()
    neuron_high.fixed_threshold = 1.0

    neuron_high.start_session(BATCH, "cpu")
    for _ in range(NUM_CHUNKS):
        h = torch.randn(BATCH, SEQ, DIMS["d_model"])
        neuron_high(h)

    high_down = torch.norm(neuron_high.W_down_mod).item()
    high_up = torch.norm(neuron_high.W_up_mod).item()
    assert high_down < 1e-2, f"W_down_mod should be near-zero (high threshold), got {high_down}"
    assert high_up < 1e-2, f"W_up_mod should be near-zero (high threshold), got {high_up}"

    # 4c: eval mode → no writes regardless
    torch.manual_seed(42)
    neuron_eval = make_neuron()
    neuron_eval.fixed_threshold = 0.0  # low threshold

    neuron_eval.start_session(BATCH, "cpu")
    neuron_eval.adapt_mode = False
    for _ in range(NUM_CHUNKS):
        h = torch.randn(BATCH, SEQ, DIMS["d_model"])
        neuron_eval(h)

    eval_down = torch.norm(neuron_eval.W_down_mod).item()
    eval_up = torch.norm(neuron_eval.W_up_mod).item()
    assert eval_down == 0.0, f"W_down_mod should be zero in eval mode, got {eval_down}"
    assert eval_up == 0.0, f"W_up_mod should be zero in eval mode, got {eval_up}"

    print(
        f"PASS: Test 4 — Projection writes: "
        f"low_thresh=(down={low_down:.4f}, up={low_up:.4f}), "
        f"high_thresh=(down={high_down:.4f}, up={high_up:.4f}), "
        f"eval_mode=(down={eval_down:.4f}, up={eval_up:.4f})"
    )


if __name__ == "__main__":
    test_output_shapes()
    test_mem_a_norm_increases()
    test_gradient_flow()
    test_projection_writes()
    print("\nAll tests passed!")
