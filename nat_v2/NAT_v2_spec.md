# NAT v2 Implementation Spec
## Minimal Architecture for Testing

---

## What This Is

An upgrade to NAT v1 (currently running, showing 0.115 adaptation benefit at episode 25500). NAT v1 has two adaptive memory layers at layers 9 and 18 with outer product writes and linear reads. NAT v2 adds three things:

1. **Attention-based memory read** (sharp selection instead of linear smear)
2. **Projection bottleneck** (learned interpretation of memory, with nonlinearity)
3. **Slow neuron with context** (nested learning — slower timescale shapes faster one)

Base model: Qwen3-4B (frozen, d_model=2560, 36 layers, bf16)
Hardware: A100 80GB

---

## Architecture Overview

Three neurons. Two fast, one slow.

```
Slow neuron (fires every 4096 tokens = 16 chunks)
    │ context (d_context=128 vector)
    │ shapes all fast neuron decisions
    ▼
Fast neuron A — hooks layer 9  (fires every 256 tokens = 1 chunk)
Fast neuron B — hooks layer 18 (fires every 256 tokens = 1 chunk)
    │ reports (d_report=128 vector per chunk)
    │ sent up after each chunk
    ▼
Slow neuron accumulates reports
```

---

## Fast Neuron — Full Pipeline

One chunk of processing at layer 9 or 18. Input: hidden state h (batch, seq, 2560) from base model, plus context (batch, d_context) from slow neuron.

### Step 1: OBSERVE

```python
# Predict what h should look like given context
predicted_h = state_predictor(prev_h_avg, context)   # θ: Linear(d_model + d_context, d_hidden) → GELU → Linear(d_hidden, d_model)
h_avg = h.mean(dim=1)                                 # (batch, d_model)
error = h_avg - predicted_h                            # (batch, d_model)
surprise = surprise_net(error, context)                # θ: Linear(d_model + d_context, d_hidden) → GELU → Linear(d_hidden, 1) → Sigmoid
                                                       # Output: (batch, 1) in [0, 1]
```

Inputs: h, prev_h_avg, context
Outputs: surprise (scalar per batch), h_avg
θ networks: state_predictor, surprise_net
State updated: prev_h_avg = h_avg.detach()

### Step 2: MEMORY WRITE

```python
write_input = cat(h_avg, surprise, context)            # (batch, d_model + 1 + d_context)
key = write_key_net(write_input)                       # θ: Linear(d_model+1+d_context, d_hidden) → GELU → Linear(d_hidden, rank)
                                                       # Output: (batch, rank)
value = write_value_net(write_input)                   # θ: Linear(d_model+1+d_context, d_hidden) → GELU → Linear(d_hidden, d_model)
                                                       # Output: (batch, d_model)
lr = lr_net(surprise, context)                         # θ: Linear(1 + d_context, d_hidden//2) → GELU → Linear(d_hidden//2, 1) → Softplus
                                                       # Output: (batch, 1), clamped to max 0.1

# Outer product write
mem_A = mem_A + lr.unsqueeze(-1) * torch.bmm(value.unsqueeze(2), key.unsqueeze(1))
# mem_A: (batch, d_model, rank) — NOT in-place, use = not +=

# Norm clamp for stability
norm = torch.norm(mem_A, dim=(1,2), keepdim=True)
mem_A = mem_A * torch.where(norm > max_norm, max_norm / (norm + 1e-8), torch.ones_like(norm))
```

Inputs: h_avg, surprise, context
Outputs: updated mem_A
θ networks: write_key_net, write_value_net, lr_net
State modified: mem_A (every chunk)

### Step 3: MEMORY READ (Attention)

```python
# Treat mem_A columns as slots to attend over
slots = mem_A.transpose(1, 2)                          # (batch, rank, d_model) — 64 slots, each 2560-dim

query = read_query_net(h_avg, context)                 # θ: Linear(d_model + d_context, d_query)
                                                       # Output: (batch, d_query)
keys = torch.bmm(slots, W_K)                           # θ: W_K is (d_model, d_query) parameter
                                                       # Output: (batch, rank, d_query)
values = torch.bmm(slots, W_V)                         # θ: W_V is (d_model, d_value) parameter
                                                       # Output: (batch, rank, d_value)

# Scaled dot-product attention
attn_weights = torch.softmax(
    torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) / sqrt(d_query),
    dim=-1
)                                                      # (batch, 1, rank)
mem_read = torch.bmm(attn_weights, values).squeeze(1)  # (batch, d_value)
```

Inputs: h_avg, context, mem_A
Outputs: mem_read (batch, d_value)
θ networks/params: read_query_net, W_K, W_V
State read: mem_A

Note: d_query = 128, d_value = 256 (or d_model, see dimensions section)

### Step 4: PROJECTION (Bottleneck with nonlinearity)

```python
# Effective weights = base + accumulated modifications
W_down = W_down_base + W_down_mod                      # (d_value, d_proj)
W_up = W_up_base + W_up_mod                            # (d_proj, d_model)

down = F.gelu(torch.bmm(mem_read.unsqueeze(1), W_down.unsqueeze(0).expand(batch,-1,-1)).squeeze(1))
                                                       # (batch, d_proj)
projected = torch.bmm(down.unsqueeze(1), W_up.unsqueeze(0).expand(batch,-1,-1)).squeeze(1)
                                                       # (batch, d_model)

# Residual: projection adds on top of raw mem_read passed through a linear map
# This ensures identity-like behavior at init
output = projected + mem_read_to_model(mem_read)       # θ: Linear(d_value, d_model) — passthrough path
```

Wait — simpler. If d_value = d_model, the residual is just mem_read itself:

```python
W_down = W_down_base + W_down_mod                      # (d_model, d_proj)
W_up = W_up_base + W_up_mod                            # (d_proj, d_model)

down = F.gelu(mem_read @ W_down)                       # (batch, d_proj) — batched, W_down shared or per-user
projected = down @ W_up                                # (batch, d_model)
output = mem_read + projected                          # residual connection
```

With W_down_base and W_up_base initialized to small values, projected ≈ 0 at init, so output ≈ mem_read. The projection learns to add corrections on top of raw recall.

**Important**: W_down and W_up need batch dimension for the mod matrices:

```python
# W_down_base: (d_model, d_proj) — θ, shared
# W_down_mod:  (batch, d_model, d_proj) — per-user state, starts at zero
# Effective per-sample:
W_down_eff = W_down_base.unsqueeze(0) + W_down_mod     # (batch, d_model, d_proj)
W_up_eff = W_up_base.unsqueeze(0) + W_up_mod           # (batch, d_proj, d_model)

down = F.gelu(torch.bmm(mem_read.unsqueeze(1), W_down_eff).squeeze(1))  # (batch, d_proj)
projected = torch.bmm(down.unsqueeze(1), W_up_eff).squeeze(1)           # (batch, d_model)
output = mem_read + projected
```

Inputs: mem_read
Outputs: output (batch, d_model)
θ parameters: W_down_base, W_up_base (initialized small random, ~0.01)
Per-user state: W_down_mod, W_up_mod (initialized zero)
Behavior at init: output ≈ mem_read (residual dominates)

### Step 5: PROJECTION WRITE (if surprise > threshold)

```python
threshold = threshold_net(context)                     # θ: Linear(d_context, 1) → Sigmoid
                                                       # Output: (batch, 1) in [0, 1]

if surprise > threshold:  # per-sample, mask for batch
    write_input = cat(h_avg, surprise, mem_read, context)
                                                       # (batch, d_model + 1 + d_model + d_context)
    raw = proj_write_net(write_input)                  # θ: Linear(in, d_hidden) → GELU → Linear(d_hidden, out)
                                                       # out = d_model + d_proj + d_proj + d_model
    
    # Split into pattern/address pairs for each matrix
    d_pat, d_addr = raw[:, :d_model], raw[:, d_model:d_model+d_proj]
    u_pat, u_addr = raw[:, d_model+d_proj:d_model+2*d_proj], raw[:, d_model+2*d_proj:]
    
    proj_lr = proj_lr_net(surprise, context)            # θ: Linear(1+d_context, 1) → Softplus
                                                       # Output: (batch, 1), clamped
    
    # Outer product writes — rank-1 updates to projection mod matrices
    # Apply surprise mask for batch elements below threshold
    mask = (surprise > threshold).float()               # (batch, 1)
    
    W_down_mod = W_down_mod + mask.unsqueeze(-1) * proj_lr.unsqueeze(-1) * torch.bmm(
        d_pat.unsqueeze(2), d_addr.unsqueeze(1)
    )                                                  # (batch, d_model, d_proj)
    
    W_up_mod = W_up_mod + mask.unsqueeze(-1) * proj_lr.unsqueeze(-1) * torch.bmm(
        u_pat.unsqueeze(2), u_addr.unsqueeze(1)
    )                                                  # (batch, d_proj, d_model)
```

Inputs: h_avg, surprise, mem_read, context
Outputs: updated W_down_mod, W_up_mod
θ networks: proj_write_net, proj_lr_net, threshold_net
State modified: W_down_mod, W_up_mod (only when surprise > threshold)

### Step 6: GATE AND INJECT

```python
g = gate_net(h_avg, output, context)                   # θ: Linear(d_model + d_model + d_context, d_hidden) → GELU → Linear(d_hidden, 1) → Sigmoid
                                                       # Initialize final bias to -1.0 (gate starts low)
                                                       # Output: (batch, 1)

# Expand gate to sequence dimension and apply
h_new = h + g.unsqueeze(1) * output.unsqueeze(1)       # (batch, seq, d_model)
h_new = layer_norm(h_new)                              # θ: LayerNorm(d_model)
```

Inputs: h, output, context
Outputs: h_new (batch, seq, d_model) — continues through base model

### Step 7: REPORT UP

```python
report = report_net(h_avg, surprise, mem_read, output)  # θ: Linear(d_model + 1 + d_model + d_model, d_hidden) → GELU → Linear(d_hidden, d_report)
                                                        # Output: (batch, d_report)
```

Inputs: h_avg, surprise, mem_read, output
Outputs: report (batch, d_report) — sent to slow neuron
θ networks: report_net

---

## Slow Neuron

The slow neuron is the same architecture applied to reports instead of hidden states. It fires every 16 chunks (4096 tokens).

### Input

Each chunk, it receives reports from both fast neurons:

```python
# Accumulate reports in a buffer
report_buffer.append(cat(report_A, report_B))          # (batch, 2 * d_report)
```

Every 16 chunks, the slow neuron fires:

```python
# Average the accumulated reports
accumulated = stack(report_buffer).mean(dim=0)          # (batch, 2 * d_report)

# The slow neuron treats this as its "h"
slow_h = accumulated
```

### Processing

The slow neuron runs the same pipeline as a fast neuron, but on slow_h instead of base model hidden states. It has its own:
- state_predictor, surprise_net (observes report patterns)
- write_key_net, write_value_net, lr_net (writes to its own mem_A)
- read_query_net, W_K, W_V (reads from its own mem_A with attention)
- W_down_base, W_up_base, W_down_mod, W_up_mod (its own projection)
- proj_write_net, proj_lr_net, threshold_net (modifies its own projection)

### Output: Context + Consolidation Writes

The slow neuron does TWO things when it fires:

**A. Produces context** (shapes how fast neurons learn):

```python
# slow neuron output after projection
slow_output = ...                                       # (batch, d_model_slow)

# Compress to context size
context = context_compress(slow_output)                 # θ: Linear(d_model_slow, d_context)
                                                       # Output: (batch, d_context)

# This context is used by fast neurons until the slow neuron fires again
fast_neuron_A.context = context
fast_neuron_B.context = context
```

**B. Writes directly to fast neuron projections** (consolidates concepts):

```python
# For each fast neuron, the slow neuron can refine its projection
# It sees: its own output + summary of fast neuron's current projection state

for fast_neuron in [fast_neuron_A, fast_neuron_B]:
    # Summarize current fast projection state
    down_norm = torch.norm(fast_neuron.W_down_mod, dim=(1,2))  # (batch,)
    up_norm = torch.norm(fast_neuron.W_up_mod, dim=(1,2))      # (batch,)
    proj_state = cat(down_norm.unsqueeze(1), up_norm.unsqueeze(1))  # (batch, 2)
    
    # Consolidation write network decides what to write
    consol_input = cat(slow_output, proj_state)         # (batch, d_model_slow + 2)
    consol_raw = consolidation_write_net(consol_input)  # θ: Linear(in, d_hidden_slow) → GELU
                                                        #    → Linear(d_hidden_slow, out)
                                                        # out = d_model + d_proj + d_proj + d_model
    
    # Split into pattern/address pairs
    d_pat = consol_raw[:, :d_model]                     # (batch, 2560) — what to write to W_down
    d_addr = consol_raw[:, d_model:d_model+d_proj]      # (batch, 128)  — where in W_down
    u_pat = consol_raw[:, d_model+d_proj:d_model+2*d_proj]  # (batch, 128) — what to write to W_up
    u_addr = consol_raw[:, d_model+2*d_proj:]           # (batch, 2560) — where in W_up
    
    consol_lr = consolidation_lr_net(slow_output)       # θ: Linear(d_model_slow, 1) → Softplus
                                                        # Output: (batch, 1), clamped small (~0.01)
    
    # Rank-1 outer product writes to fast neuron projection
    fast_neuron.W_down_mod = fast_neuron.W_down_mod + consol_lr.unsqueeze(-1) * torch.bmm(
        d_pat.unsqueeze(2), d_addr.unsqueeze(1)
    )
    fast_neuron.W_up_mod = fast_neuron.W_up_mod + consol_lr.unsqueeze(-1) * torch.bmm(
        u_pat.unsqueeze(2), u_addr.unsqueeze(1)
    )
```

This is the consolidation loop:
1. Fast neurons discover concepts through experience (projection writes from surprise)
2. Fast neurons report what happened to the slow neuron
3. Slow neuron stores and interprets reports over many chunks
4. When the slow neuron fires, it refines the fast neuron's projection based on its longer-term view
5. Concepts flow up as reports, get interpreted, and flow back down as direct weight modifications

The consolidation_write_net and consolidation_lr_net are θ — trained during Phase 2.
The consol_lr is clamped small so the slow neuron makes gentle refinements, not dramatic rewrites.

**Additional slow neuron θ networks (trained in Phase 2):**
```
consolidation_write_net:  (d_model_slow + 2) → d_hidden_slow → (d_model + d_proj + d_proj + d_model) = ~2.5M
consolidation_lr_net:     d_model_slow → 1                                                          = ~256
```
One consolidation_write_net per fast neuron, or shared with a fast-neuron-id input.

### Slow Neuron's Own Context

For this experiment, the slow neuron receives no context from above (no glacial neuron). Its context input is a learned constant:

```python
slow_context = slow_default_context                     # θ: nn.Parameter(d_context), learned
```

If this works, we add the glacial neuron later.

### Slow Neuron d_model

The slow neuron operates on reports, not hidden states. Its internal d_model is 2 * d_report = 256, not 2560. This makes it much smaller and cheaper.

---

## Dimensions

```
d_model     = 2560   (Qwen3-4B hidden size)
rank        = 64     (memory slots)
d_query     = 128    (attention query dimension)
d_value     = 256    (attention value dimension — what gets read out)
d_proj      = 128    (projection bottleneck width)
d_context   = 128    (context vector from slow neuron)
d_report    = 128    (report vector sent to slow neuron)
d_hidden    = 384    (hidden dim for all θ MLPs)
max_norm    = 10.0   (memory norm clamp)

Slow neuron dimensions:
d_model_slow = 256   (2 * d_report, its input)
rank_slow    = 32    (smaller memory)
d_query_slow = 64
d_value_slow = 128
d_proj_slow  = 64
d_hidden_slow = 192
```

Note: d_value is the dimension of the memory read output. Since we have a projection bottleneck after the read, d_value does NOT need to be d_model. It can be smaller (256), and the W_up in the projection maps from d_proj to d_model. But the residual path needs mem_read to be d_model for `output = mem_read + projected`.

**Revised**: set d_value = d_model = 2560 so the residual works cleanly. The projection bottleneck (d_proj=128) provides the compression. The attention values are full d_model.

```
REVISED:
d_value = 2560 (= d_model, for clean residual)
W_down_base: (2560, 128)
W_up_base:   (128, 2560)
```

---

## Per-User State (persisted between sessions)

### Fast Neuron (× 2 layers):
```
mem_A:        (d_model, rank) = (2560, 64)     — 655 KB (bf16)
W_down_mod:   (d_model, d_proj) = (2560, 128)  — 655 KB (bf16)
W_up_mod:     (d_proj, d_model) = (128, 2560)  — 655 KB (bf16)
prev_h_avg:   (d_model,) = (2560,)             — 5 KB (bf16)
```
Per fast neuron: ~2 MB
Two fast neurons: ~4 MB

### Slow Neuron (× 1):
```
mem_A:        (256, 32)                         — 16 KB
W_down_mod:   (256, 64)                         — 33 KB
W_up_mod:     (64, 256)                         — 33 KB
prev_h_avg:   (256,)                            — 0.5 KB
current_context: (d_context,) = (128,)          — 0.25 KB
```
Slow neuron: ~83 KB

**Total per-user state: ~4.1 MB**

---

## θ Parameter Count (trained, frozen at deployment)

### Per Fast Neuron:
```
state_predictor:    (2560+128) → 384 → 2560     = ~2.0M
surprise_net:       (2560+128) → 384 → 1        = ~1.0M
write_key_net:      (2560+1+128) → 384 → 64     = ~1.1M
write_value_net:    (2560+1+128) → 384 → 2560   = ~2.1M
lr_net:             (1+128) → 192 → 1            = ~25K
read_query_net:     (2560+128) → 128             = ~344K
W_K:                (2560, 128)                  = ~328K
W_V:                (2560, 2560)                 = ~6.6M
W_down_base:        (2560, 128)                  = ~328K
W_up_base:          (128, 2560)                  = ~328K
proj_write_net:     (2560+1+2560+128) → 384 → (2560+128+128+2560) = ~4.0M
proj_lr_net:        (1+128) → 64 → 1            = ~8K
threshold_net:      128 → 1                      = ~128
gate_net:           (2560+2560+128) → 384 → 1   = ~2.0M
report_net:         (2560+1+2560+2560) → 384 → 128 = ~3.0M
layer_norm:         2560                         = ~5K
```
Per fast neuron: ~23M parameters

### Slow Neuron:
Same structure, smaller dimensions. ~1.5M parameters.

### Total θ: ~47.5M (two fast + one slow)

---

## Memory Budget (A100 80GB)

```
Base model (4B, bf16):                     ~8 GB
θ parameters (47.5M, fp32):               ~190 MB
Per-user state (batch 4):                  ~16 MB
BPTT chain (8 chunks × activations):      ~20 GB
Base model activations:                    ~20 GB
Optimizer (AdamW on 47.5M):               ~380 MB
PyTorch overhead:                          ~5 GB
Total:                                     ~54 GB
```

Fits with headroom.

---

## Training

### Phase 1: Within-Session Adaptation (50000 episodes)

Trains ALL fast neuron θ. Slow neuron INACTIVE (context = zeros).

**Episode structure:**
- 2048 tokens from one domain
- Split into 8 chunks of 256 tokens
- Chunks 1-6: adapt (memory writes, projection writes, read, gate)
- Chunks 7-8: evaluate (read only, no writes)
- Loss: cross-entropy on eval chunks only

**What gets trained:**
Everything in both fast neurons. state_predictor, surprise_net, write_key_net, write_value_net, lr_net, read_query_net, W_K, W_V, W_down_base, W_up_base, proj_write_net, proj_lr_net, threshold_net, gate_net, report_net, layer_norm.

**Key training details:**
- Optimizer: AdamW, lr=3e-4, weight_decay=0.01
- No KV cache (each chunk independent, fast weights are cross-chunk memory)
- enable_thinking=False for Qwen3
- Batch dimension on ALL state tensors: mem_A is (batch, d_model, rank)
- Use = not += for all state updates (autograd compatibility)
- BPTT through all 6 adapt chunks per episode
- Detach state at episode boundaries (start_session resets)
- Gradient clipping: max_norm=1.0

**Success metric:** adaptation_benefit = loss_chunk1 - loss_chunk8 > 0 and increasing

**Datasets:** AMPS math, MATH hard, CodeForces-CoTs, AR-LSAT, DROP, ScienceQA. Same as NAT v1.

### Phase 2: Across-Session Consolidation (5000 multi-window episodes)

Activates slow neuron. Trains slow neuron θ + context pathways + consolidation writes.

**Episode structure:**
- 13 windows of 2048 tokens each (26624 tokens total)
- Windows 1-5: domain A
- Windows 6-10: domain B
- Windows 11-13: domain A again (forgetting test)
- Fast neuron memory resets each window
- Slow neuron memory persists across all windows
- Slow neuron fires after every 16 chunks (2 firings per window)
- Projection mod matrices persist across all windows (this is where concepts live)
- On each slow neuron firing: context update + consolidation writes to fast projections

**What gets trained:**
- Slow neuron: all its θ networks (observe, write, read, project, report)
- Slow neuron: consolidation_write_net, consolidation_lr_net (the direct projection writes)
- Fast neurons: context pathways refined (lr=1e-5 for fast θ, full lr for slow θ)

**The consolidation loop being trained:**
1. Fast neurons adapt within each window (memory writes + projection writes from surprise)
2. Fast neurons send reports to slow neuron each chunk
3. Slow neuron fires every 16 chunks, producing:
   a. New context → shapes fast neuron behavior in subsequent chunks
   b. Consolidation writes → directly refines fast neuron projection weights
4. When domain switches (window 6), fast memory resets but projection mods persist
5. Slow neuron's consolidation writes should have captured domain A concepts in the projection
6. When domain A returns (window 11), projection already contains consolidated knowledge

**Loss:**
- Primary: cross-entropy on eval chunks within each window
- Forgetting penalty: loss on domain A windows 11-13 should be ≤ loss on domain A windows 1-5

**Success metric:** forgetting_ratio = loss_A_after / loss_A_before < 1.0. Adaptation benefit maintained. Consolidation writes have nonzero norm.

---

## Implementation Order

1. **Implement FastNeuron class** — the full pipeline from observe to report
2. **Implement SlowNeuron class** — same pipeline, different dims, fires every 16 chunks
3. **Implement NATv2Model** — hooks into Qwen3-4B, manages neurons, state, context flow
4. **Implement Phase 1 training loop** — episodes, BPTT, logging
5. **Run Phase 1** — validate adaptation benefit > 0
6. **Implement Phase 2 training loop** — multi-window episodes, slow neuron active
7. **Run Phase 2** — validate forgetting decreases

---

## Critical Implementation Details

### Hook-Based Architecture

Do NOT manually iterate through base model layers. Use PyTorch forward hooks:

```python
def make_hook(neuron):
    def hook_fn(module, input, output):
        h = output[0]  # hidden states
        h_new = neuron(h)
        return (h_new,) + output[1:]
    return hook_fn

model.model.layers[9].register_forward_hook(make_hook(fast_neuron_A))
model.model.layers[18].register_forward_hook(make_hook(fast_neuron_B))
```

### State Management

```python
class FastNeuron(nn.Module):
    def start_session(self, batch_size, device):
        """Reset all per-user state for a new episode."""
        self.mem_A = torch.zeros(batch_size, self.d_model, self.rank, device=device)
        self.W_down_mod = torch.zeros(batch_size, self.d_model, self.d_proj, device=device)
        self.W_up_mod = torch.zeros(batch_size, self.d_proj, self.d_model, device=device)
        self.prev_h_avg = None
        self.context = torch.zeros(batch_size, self.d_context, device=device)
```

### No In-Place Operations

Every state update must create a new tensor:
```python
# CORRECT:
self.mem_A = self.mem_A + delta

# WRONG:
self.mem_A += delta
```

### Chunk Processing

Each chunk of 256 tokens is processed independently through the base model. No KV cache across chunks. The fast weights (mem_A, W_down_mod, W_up_mod) ARE the cross-chunk memory.

```python
for chunk_idx in range(num_chunks):
    chunk_ids = input_ids[:, chunk_idx*256:(chunk_idx+1)*256]
    
    # Forward through base model (hooks fire automatically)
    outputs = model(chunk_ids)
    
    if chunk_idx < num_adapt_chunks:
        loss = compute_loss(outputs, chunk_ids)
        # Don't step optimizer — just accumulate into BPTT chain
    else:
        eval_loss += compute_loss(outputs, chunk_ids)

# After all chunks: backprop through entire chain
eval_loss.backward()
optimizer.step()
```

### Slow Neuron Firing

```python
chunk_counter = 0
for chunk_idx in range(num_chunks):
    # ... process chunk ...
    
    # Collect reports
    report_A = fast_neuron_A.last_report
    report_B = fast_neuron_B.last_report
    slow_neuron.accumulate_report(cat(report_A, report_B))
    
    chunk_counter += 1
    if chunk_counter % 16 == 0:
        # Slow neuron fires
        new_context = slow_neuron.fire()
        fast_neuron_A.context = new_context
        fast_neuron_B.context = new_context
```

### W_V Size Concern

W_V at (2560, 2560) is 6.6M params per fast neuron — the single largest θ component. Alternative: use d_value = 512 and add a linear projection from 512 → 2560 before the residual. This cuts W_V to (2560, 512) = 1.3M and adds a (512, 2560) projection = 1.3M. Total 2.6M instead of 6.6M. The projection is θ.

**Recommended**: Use d_value = 512 with a value_proj:

```python
values = torch.bmm(slots, W_V)                        # (batch, rank, 512)
mem_read_compressed = torch.bmm(attn_weights, values)  # (batch, 512)
mem_read = value_up_proj(mem_read_compressed)           # θ: Linear(512, 2560) → (batch, 2560)
```

This saves ~8M parameters across both fast neurons.

---

## File Structure

```
nat_v2/
├── model/
│   ├── __init__.py
│   ├── fast_neuron.py          # FastNeuron class — full pipeline
│   ├── slow_neuron.py          # SlowNeuron class — same pipeline, different dims
│   ├── nat_model.py            # NATv2Model — hooks, state management, context flow
│   └── utils.py                # Helpers
├── training/
│   ├── __init__.py
│   ├── phase1_adaptation.py    # Phase 1: within-session adaptation
│   ├── phase2_consolidation.py # Phase 2: across-session with slow neuron
│   ├── data.py                 # Episode construction
│   └── eval.py                 # Evaluation
├── configs/
│   └── base.yaml
├── scripts/
│   ├── train_phase1.py
│   ├── train_phase2.py
│   └── evaluate.py
└── README.md
```

---

## What Makes This Different From NAT v1

| | NAT v1 | NAT v2 |
|---|---|---|
| Memory read | Linear (mem_A @ query) | Attention over slots (sharp selection) |
| Post-read processing | None | Projection bottleneck with GELU + residual |
| Projection adaptation | N/A | Outer product writes to W_down_mod, W_up_mod |
| Slow neuron | None | Fires every 16 chunks, produces context |
| Context | None | Shapes observe, write, read, projection, gate |
| Per-user state | mem_A only (~655KB) | mem_A + W_down_mod + W_up_mod (~2MB per layer) |
| Continual learning | None (state resets) | Projection mods persist, accumulate across sessions |
| New thought generation | No (raw recall only) | Yes (nonlinear projection + base model reasoning) |

---

## Quick Validation Test

Before full training, run this sanity check:

```python
# 1. Forward pass produces valid outputs
# 2. Adaptation benefit > 0 within 100 episodes
# 3. Projection gate opens (> 0.1) within 500 episodes
# 4. Surprise values are in reasonable range [0.1, 0.9]
# 5. Memory norm stays below max_norm
# 6. W_down_mod and W_up_mod norms grow slowly (< 1.0 after 1000 episodes)
# 7. Slow neuron context is non-zero after first firing (Phase 2 only)
```

---

## Research Questions This Experiment Answers

1. Does attention-based memory read outperform linear read? (Compare v2 Phase 1 vs v1)
2. Does the projection bottleneck add value beyond raw recall? (Monitor projection gate)
3. Does the nonlinearity matter? (Ablation: remove GELU, compare)
4. Does slow neuron context improve fast neuron adaptation? (Compare Phase 2 with/without context)
5. Do projection mods accumulate meaningfully? (Monitor W_mod norms and projection output divergence from identity)
6. Does knowledge persist across sessions? (Save state, start new session, measure initial performance)
7. Do consolidation writes help preserve knowledge across domain switches? (Compare forgetting with/without consolidation writes)
8. Does the concept flow loop work? (Fast discovers → reports up → slow consolidates → writes down → fast retains)

---

## Concept Flow Summary

```
DISCOVERY (fast, every chunk):
  Surprise → projection write → new pattern detector in W_mod
  "I learned that X+Y means Z in this user's domain"

REPORTING (fast → slow, every chunk):
  Report encodes what happened, including projection's contribution
  "Here's what I saw, how surprised I was, what I recalled, what I produced"

INTERPRETATION (slow, every 16 chunks):
  Slow neuron stores reports, reads patterns across many chunks
  "The fast neurons keep encountering this type of pattern"

CONSOLIDATION (slow → fast, every 16 chunks):
  Context: shapes how fast neurons observe, write, read, project
  "Pay more attention to X, write more aggressively when Y"
  
  Direct writes: refines fast neuron projection weights
  "Reinforce the concept you discovered, it's real and persistent"

PERSISTENCE (across sessions):
  All projection mod matrices (fast + slow) saved and restored
  Concepts survive session boundaries
  Slow neuron memory also persists — long-term behavioral patterns retained
```