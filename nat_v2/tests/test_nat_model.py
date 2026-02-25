"""
Tests for NATv2Model — integration tests with a Qwen3 base model.

Uses a tiny Qwen3 model (4 layers, d_model=64) for fast CPU testing.
Set env NAT_FULL_MODEL=1 to test with the real Qwen3-4B on GPU.

Run with: python -m tests.test_nat_model
"""

import os
import tempfile

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from model.nat_model import NATv2Model

# ── Tiny Qwen3 config for fast CPU testing ──────────────────────────
TINY_D_MODEL = 64
TINY_LAYERS = 4
TINY_VOCAB = 256

BATCH = 2
SEQ = 16
NUM_ADAPT = 3
NUM_EVAL = 1
NUM_CHUNKS = NUM_ADAPT + NUM_EVAL


def make_tiny_base_model():
    """Create a tiny Qwen3 model (~0.16M params) for testing."""
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-4B", local_files_only=True,
    )
    config.hidden_size = TINY_D_MODEL
    config.num_hidden_layers = TINY_LAYERS
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.intermediate_size = 128
    config.vocab_size = TINY_VOCAB
    config.head_dim = TINY_D_MODEL // 4
    return AutoModelForCausalLM.from_config(config)


def make_model():
    """Create NATv2Model with the tiny base model, hooks at layers 1 and 2."""
    base = make_tiny_base_model()
    # Hooks at layers 1 and 2 (tiny model only has 4 layers)
    model = NATv2Model(
        base_model=base,
        layer_A=1,
        layer_B=2,
        dtype=torch.float32,
    )
    return model


def make_input():
    return torch.randint(0, TINY_VOCAB, (BATCH, SEQ))


# ── Test 1: Model creation ──────────────────────────────────────────
def test_creation():
    model = make_model()

    expected_hooks = 1  # neuron A disabled by default
    assert len(model._hook_handles) == expected_hooks, \
        f"Expected {expected_hooks} hooks, got {len(model._hook_handles)}"

    # Base model frozen, neurons trainable
    base_trainable = sum(1 for p in model.base_model.parameters() if p.requires_grad)
    theta_count = model.count_theta_params()
    assert base_trainable == 0, f"Base model should be frozen, {base_trainable} params trainable"
    assert theta_count > 0, "Should have trainable θ parameters"

    print(f"PASS: Test 1 — Model created: {theta_count:,} θ params, {expected_hooks} hook(s)")


# ── Test 7: Output shapes match base model ──────────────────────────
def test_output_shapes():
    torch.manual_seed(0)
    model = make_model()
    model.start_episode(BATCH, "cpu")

    ids = make_input()
    with torch.no_grad():
        out = model(ids, adapt=False)

    assert out.logits.shape == (BATCH, SEQ, TINY_VOCAB), \
        f"Logits shape: {out.logits.shape}"

    print(f"PASS: Test 7 — Output shape correct: {out.logits.shape}")


# ── Test 4: Neurons are identity when mem_A is zero ─────────────────
def test_identity_when_empty():
    torch.manual_seed(0)
    model = make_model()
    ids = make_input()

    # Fresh session → mem_A is zeros → early exit → pure identity
    model.start_episode(BATCH, "cpu")

    with torch.no_grad():
        out_with_hooks = model(ids, adapt=False)

    # Reset state again (the forward above set prev_h_avg)
    model.start_episode(BATCH, "cpu")

    # Forward without hooks → pure frozen baseline
    model.remove_hooks()
    with torch.no_grad():
        out_without_hooks = model.base_model(ids, use_cache=False)
    model.register_hooks()

    assert torch.equal(out_with_hooks.logits, out_without_hooks.logits), \
        "Identity check failed: hooks with zero mem_A should produce identical output"

    print("PASS: Test 4 — Neurons are identity when mem_A is zero")


# ── Test 5: mem_A changes after first adapt step ────────────────────
def test_mem_a_changes():
    torch.manual_seed(0)
    model = make_model()
    model.start_episode(BATCH, "cpu")

    norm_B_before = torch.norm(model.fast_neuron_B.mem_A).item()
    assert norm_B_before == 0.0

    ids = make_input()
    with torch.no_grad():
        model(ids, adapt=True)

    norm_B_after = torch.norm(model.fast_neuron_B.mem_A).item()
    assert norm_B_after > 0, f"fast_B mem_A should be nonzero after adapt, got {norm_B_after}"

    print(
        f"PASS: Test 5 — mem_A changes: "
        f"B={norm_B_before:.4f}→{norm_B_after:.4f}"
    )


# ── Test 2: Run 4 chunks with adaptation ────────────────────────────
def test_multi_chunk_adaptation():
    torch.manual_seed(0)
    model = make_model()
    model.start_episode(BATCH, "cpu")

    norms_B = []
    for i in range(NUM_CHUNKS):
        ids = make_input()
        adapt = i < NUM_ADAPT
        with torch.no_grad():
            out = model(ids, adapt=adapt)
        norms_B.append(torch.norm(model.fast_neuron_B.mem_A).item())

    # Memory should grow over adapt chunks
    assert norms_B[-1] > norms_B[0], f"fast_B mem_A should grow: {norms_B}"

    # Reports should have been collected
    assert len(model.slow_neuron.report_buffer) == NUM_CHUNKS, \
        f"Expected {NUM_CHUNKS} reports, got {len(model.slow_neuron.report_buffer)}"
    assert model.chunk_counter == NUM_CHUNKS

    print(
        f"PASS: Test 2 — 4-chunk adaptation: "
        f"B_norms={[f'{n:.4f}' for n in norms_B]}"
    )


# ── Test 3: Frozen baseline is truly frozen ─────────────────────────
def test_frozen_baseline():
    torch.manual_seed(0)
    model = make_model()
    model.start_episode(BATCH, "cpu")

    ids = make_input()

    # First: run adapt chunks so neurons have memory content
    for _ in range(NUM_ADAPT):
        with torch.no_grad():
            model(make_input(), adapt=True)

    # Frozen baseline 1 (hooks removed)
    model.remove_hooks()
    with torch.no_grad():
        frozen_1 = model.base_model(ids, use_cache=False).logits.clone()
    model.register_hooks()

    # Adapted forward (hooks active, eval mode — reads from memory)
    with torch.no_grad():
        adapted = model(ids, adapt=False).logits.clone()

    # Frozen baseline 2 (hooks removed again — should match frozen_1)
    model.remove_hooks()
    with torch.no_grad():
        frozen_2 = model.base_model(ids, use_cache=False).logits.clone()
    model.register_hooks()

    # Frozen baselines should be identical
    assert torch.equal(frozen_1, frozen_2), \
        "Frozen baseline should be deterministic after hook remove/register"

    # Adapted output should differ from frozen (neurons inject memory content)
    diff = (adapted - frozen_1).abs().sum().item()
    assert diff > 0, \
        "Adapted output should differ from frozen baseline"

    print(f"PASS: Test 3 — Frozen baseline deterministic, adapted diff={diff:.4f}")


# ── Test 6: Gradients flow from eval loss to all θ networks ─────────
def test_gradient_flow():
    torch.manual_seed(0)
    model = make_model()

    # Force low thresholds so projection write paths are active
    model.fast_neuron_B.fixed_threshold = 0.0

    model.start_episode(BATCH, "cpu")
    model.zero_grad()

    # Run adapt chunks (gradients accumulate through mem_A state)
    for i in range(NUM_ADAPT):
        ids = make_input()
        model(ids, adapt=True)

    # Run eval chunk — loss computed here
    eval_ids = make_input()
    out = model(eval_ids, adapt=False)

    # Cross-entropy loss on eval chunk (shift by 1 for next-token prediction)
    logits = out.logits[:, :-1].reshape(-1, TINY_VOCAB)
    targets = eval_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # Check gradient flow to neuron B (neuron A is disabled)
    # report_net doesn't affect logits (feeds slow neuron, inactive in Phase 1).
    # It gets gradients in Phase 2 when slow neuron context feeds back to loss.
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
    ]

    neuron = model.fast_neuron_B
    params_with_grad = []
    params_no_grad = []
    for name, p in neuron.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            params_with_grad.append(name)
        else:
            params_no_grad.append(name)

    missing = []
    for prefix in critical_prefixes:
        if not any(n.startswith(prefix) for n in params_with_grad):
            missing.append(prefix)

    assert not missing, \
        f"fast_neuron_B: no gradient for {missing}. " \
        f"With grad: {params_with_grad}"

    total = len(params_with_grad) + len(params_no_grad)
    print(
        f"  fast_neuron_B: {len(params_with_grad)}/{total} params with grad"
    )
    if params_no_grad:
        print(f"    (no grad: {params_no_grad})")

    print("PASS: Test 6 — Gradients flow to all θ networks in neuron B")


# ── Test: save/load state roundtrip ─────────────────────────────────
def test_save_load_state():
    torch.manual_seed(0)
    model = make_model()
    model.start_episode(BATCH, "cpu")

    # Run some chunks to populate state
    for _ in range(3):
        with torch.no_grad():
            model(make_input(), adapt=True)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    model.save_state(path)

    # Record state before load
    mem_B_before = model.fast_neuron_B.mem_A.clone()
    counter_before = model.chunk_counter

    # Corrupt state
    model.start_episode(BATCH, "cpu")
    assert torch.norm(model.fast_neuron_B.mem_A).item() == 0.0

    # Load
    model.load_state(path)

    assert torch.equal(model.fast_neuron_B.mem_A, mem_B_before), \
        "mem_A should match after load"
    assert model.chunk_counter == counter_before, \
        "chunk_counter should match after load"

    os.unlink(path)
    print("PASS: Test — save/load state roundtrip")


# ── Test: hook remove/register lifecycle ────────────────────────────
def test_hook_lifecycle():
    model = make_model()

    assert len(model._hook_handles) == 1  # neuron A disabled
    model.remove_hooks()
    assert len(model._hook_handles) == 0

    # Double remove is safe
    model.remove_hooks()
    assert len(model._hook_handles) == 0

    model.register_hooks()
    assert len(model._hook_handles) == 1

    # Double register doesn't add extra hooks
    model.register_hooks()
    assert len(model._hook_handles) == 1

    print("PASS: Test — Hook lifecycle (remove/register idempotent)")


if __name__ == "__main__":
    test_creation()
    test_output_shapes()
    test_identity_when_empty()
    test_mem_a_changes()
    test_multi_chunk_adaptation()
    test_frozen_baseline()
    test_gradient_flow()
    test_save_load_state()
    test_hook_lifecycle()
    print("\nAll tests passed!")
