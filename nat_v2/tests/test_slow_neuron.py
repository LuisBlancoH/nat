"""
Tests for SlowNeuron — verifies shapes, report accumulation, consolidation
writes, gradient flow, and multi-firing memory growth.

Uses small dimensions for speed. Run with: python -m tests.test_slow_neuron
"""

import torch

from model.fast_neuron import FastNeuron
from model.slow_neuron import SlowNeuron

# Small dims — must be consistent between fast and slow
FAST_DIMS = dict(
    d_model=64, rank=8, d_query=16, d_value=32,
    d_proj=16, d_context=16, d_report=16, d_hidden=32,
)
SLOW_DIMS = dict(
    d_model_slow=32,  # 2 * d_report
    rank=4, d_query=8, d_value=16, d_proj=8,
    d_context=16, d_hidden=24,
    fast_d_model=64, fast_d_proj=16, num_fast_neurons=2,
)
BATCH = 2
NUM_REPORTS = 4  # accumulate before firing


def make_neurons():
    fast_a = FastNeuron(**FAST_DIMS)
    fast_b = FastNeuron(**FAST_DIMS)
    slow = SlowNeuron(**SLOW_DIMS)
    return fast_a, fast_b, slow


def fake_reports(n, batch=BATCH, d_report=FAST_DIMS["d_report"]):
    """Generate n fake combined reports (batch, 2*d_report)."""
    return [torch.randn(batch, 2 * d_report) for _ in range(n)]


# ------------------------------------------------------------------
# Test 1: fire() output shape and report buffer clearing
# ------------------------------------------------------------------
def test_fire_shapes_and_buffer():
    torch.manual_seed(0)
    fast_a, fast_b, slow = make_neurons()
    fast_a.start_session(BATCH, "cpu")
    fast_b.start_session(BATCH, "cpu")
    slow.start_session(BATCH, "cpu")

    for r in fake_reports(NUM_REPORTS):
        slow.accumulate_report(r)

    assert len(slow.report_buffer) == NUM_REPORTS

    ctx = slow.fire([fast_a, fast_b])

    assert ctx.shape == (BATCH, SLOW_DIMS["d_context"]), \
        f"context shape: {ctx.shape}"
    assert len(slow.report_buffer) == 0, "buffer should be cleared after fire"

    print("PASS: Test 1 — fire() shape correct, buffer cleared")


# ------------------------------------------------------------------
# Test 2: Consolidation writes modify fast neuron W_mod
# ------------------------------------------------------------------
def test_consolidation_writes():
    torch.manual_seed(0)
    fast_a, fast_b, slow = make_neurons()
    fast_a.start_session(BATCH, "cpu")
    fast_b.start_session(BATCH, "cpu")
    slow.start_session(BATCH, "cpu")

    # Record pre-fire norms (should be zero)
    pre_a_down = torch.norm(fast_a.W_down_mod).item()
    pre_b_down = torch.norm(fast_b.W_down_mod).item()
    assert pre_a_down == 0.0 and pre_b_down == 0.0

    for r in fake_reports(NUM_REPORTS):
        slow.accumulate_report(r)
    slow.fire([fast_a, fast_b])

    post_a_down = torch.norm(fast_a.W_down_mod).item()
    post_a_up = torch.norm(fast_a.W_up_mod).item()
    post_b_down = torch.norm(fast_b.W_down_mod).item()
    post_b_up = torch.norm(fast_b.W_up_mod).item()

    assert post_a_down > 0, f"fast_a W_down_mod should be modified, got {post_a_down}"
    assert post_a_up > 0, f"fast_a W_up_mod should be modified, got {post_a_up}"
    assert post_b_down > 0, f"fast_b W_down_mod should be modified, got {post_b_down}"
    assert post_b_up > 0, f"fast_b W_up_mod should be modified, got {post_b_up}"

    print(
        f"PASS: Test 2 — Consolidation writes: "
        f"A=(down={post_a_down:.6f}, up={post_a_up:.6f}), "
        f"B=(down={post_b_down:.6f}, up={post_b_up:.6f})"
    )


# ------------------------------------------------------------------
# Test 3: Slow neuron mem_A grows across multiple firings
# ------------------------------------------------------------------
def test_mem_a_growth():
    torch.manual_seed(0)
    fast_a, fast_b, slow = make_neurons()
    fast_a.start_session(BATCH, "cpu")
    fast_b.start_session(BATCH, "cpu")
    slow.start_session(BATCH, "cpu")

    norms = []
    for _ in range(4):
        for r in fake_reports(NUM_REPORTS):
            slow.accumulate_report(r)
        slow.fire([fast_a, fast_b])
        norms.append(torch.norm(slow.mem_A).item())

    assert norms[0] > 0, f"mem_A should be nonzero after first firing: {norms[0]}"
    assert norms[-1] > norms[0], f"mem_A norm should grow: {norms}"

    print(f"PASS: Test 3 — Slow mem_A grows: {[f'{n:.4f}' for n in norms]}")


# ------------------------------------------------------------------
# Test 4: Gradients flow through the entire chain
# ------------------------------------------------------------------
def test_gradient_flow():
    torch.manual_seed(0)
    fast_a, fast_b, slow = make_neurons()

    # Force thresholds low so projection write path is active
    with torch.no_grad():
        slow.threshold_net[0].bias.fill_(-10.0)

    fast_a.start_session(BATCH, "cpu")
    fast_b.start_session(BATCH, "cpu")
    slow.start_session(BATCH, "cpu")

    # Fire TWICE so proj_write_net gets gradients: first fire writes to
    # slow.W_mod, second fire reads from it through the projection step.
    slow.zero_grad()
    all_reports = []
    for _ in range(2):
        reports = [torch.randn(BATCH, 2 * FAST_DIMS["d_report"], requires_grad=True)
                   for _ in range(NUM_REPORTS)]
        all_reports.extend(reports)
        for r in reports:
            slow.accumulate_report(r)
        ctx = slow.fire([fast_a, fast_b])

    # Loss on context + fast neuron W_mod (exercises consolidation path)
    loss = (
        ctx.sum()
        + fast_a.W_down_mod.sum() + fast_a.W_up_mod.sum()
        + fast_b.W_down_mod.sum() + fast_b.W_up_mod.sum()
    )
    loss.backward()

    # Check gradient flow to slow neuron params
    params_with_grad = []
    params_no_grad = []
    for name, p in slow.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            params_with_grad.append(name)
        else:
            params_no_grad.append(name)

    critical_prefixes = [
        "default_context",
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
        "proj_write_net",
        "proj_lr_net",
        "context_compress",
        "consolidation_write_nets",
        "consolidation_lr_net",
    ]
    missing = []
    for prefix in critical_prefixes:
        if not any(n.startswith(prefix) or n == prefix for n in params_with_grad):
            missing.append(prefix)

    assert not missing, \
        f"No gradient for: {missing}. With grad: {params_with_grad}"

    total = len(params_with_grad) + len(params_no_grad)
    print(f"PASS: Test 4 — Gradients flow to {len(params_with_grad)}/{total} params")
    if params_no_grad:
        print(f"  (No grad: {params_no_grad})")

    # Also verify gradients reach the input reports
    reports_with_grad = sum(1 for r in all_reports if r.grad is not None and r.grad.abs().sum() > 0)
    assert reports_with_grad > 0, "Gradients should reach input reports"
    print(f"  Gradients reach {reports_with_grad}/{len(all_reports)} input reports")


# ------------------------------------------------------------------
# Test 5: Context changes across firings (not stuck at constant)
# ------------------------------------------------------------------
def test_context_varies():
    torch.manual_seed(0)
    fast_a, fast_b, slow = make_neurons()
    fast_a.start_session(BATCH, "cpu")
    fast_b.start_session(BATCH, "cpu")
    slow.start_session(BATCH, "cpu")

    contexts = []
    for _ in range(3):
        for r in fake_reports(NUM_REPORTS):
            slow.accumulate_report(r)
        ctx = slow.fire([fast_a, fast_b])
        contexts.append(ctx.detach().clone())

    # Contexts from different firings should differ (memory evolves)
    diff_01 = (contexts[0] - contexts[1]).abs().sum().item()
    diff_12 = (contexts[1] - contexts[2]).abs().sum().item()
    assert diff_01 > 0, "Context should change between firings"
    assert diff_12 > 0, "Context should keep changing"

    print(f"PASS: Test 5 — Context varies: diff01={diff_01:.4f}, diff12={diff_12:.4f}")


if __name__ == "__main__":
    test_fire_shapes_and_buffer()
    test_consolidation_writes()
    test_mem_a_growth()
    test_gradient_flow()
    test_context_varies()
    print("\nAll tests passed!")
