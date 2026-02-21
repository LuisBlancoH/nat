# Implementation Prompt: Nested Adaptive Transformer (NAT)

## Instructions for the Coding Agent

You are building a research prototype of the Nested Adaptive Transformer — a self-modifying transformer architecture that learns at inference time. This is a novel architecture. There is no existing codebase to reference. You are building it from scratch on top of HuggingFace Transformers and PyTorch.

Read this entire document before writing any code. Understand the architecture first, then implement in the order specified.

---

## What You Are Building

A frozen pretrained transformer with three additional modules inserted into its forward pass:

1. **Two Adaptive Memory Layers** — small modules that modify their own weights during inference, learning from the model's hidden states in real time.
2. **One Consolidation Layer** — a slow-learning module that accumulates persistent knowledge across sessions via exponential moving average.

The key idea: the adaptive layers have two kinds of parameters:
- **Slow parameters θ** — define HOW to learn. These are trained once (meta-learning) and then frozen forever.
- **Fast weights W** — the actual learned knowledge. These change during every forward pass based on the learning rule defined by θ. They are never directly trained by backprop — they emerge from the self-modification process.

Training teaches θ (the learning rule). Inference uses θ to drive self-modification of W. The model learns at inference time without any retraining.

---

## Tech Stack

```
Python 3.10+
PyTorch 2.1+
transformers (HuggingFace)
datasets (HuggingFace)
accelerate
wandb (logging)
```

Base model: `Qwen/Qwen2.5-1.5B` (or `meta-llama/Llama-3.2-1B` as fallback)

Install:
```bash
pip install torch transformers datasets accelerate wandb einops
```

---

## Project Structure

```
nat/
├── model/
│   ├── __init__.py
│   ├── adaptive_layer.py        # The adaptive memory layer
│   ├── consolidation_layer.py   # The consolidation layer
│   ├── nat_model.py             # Full model wrapping frozen base + adaptive + consolidation
│   └── utils.py                 # Helpers (low-rank ops, gradient checkpointing)
├── training/
│   ├── __init__.py
│   ├── phase1_meta_learn.py     # Phase 1: meta-learn the learning rule
│   ├── phase2_episodic.py       # Phase 2: multi-task episodic training
│   ├── phase3_consolidation.py  # Phase 3: consolidation dynamics
│   ├── data.py                  # Data loading and episode construction
│   └── eval.py                  # Evaluation and benchmarking
├── inference/
│   ├── __init__.py
│   ├── session.py               # Session management (save/load consolidation state)
│   └── generate.py              # Text generation with self-modification active
├── configs/
│   ├── base.yaml                # Default hyperparameters
│   └── small.yaml               # Small-scale config for debugging
├── scripts/
│   ├── train_phase1.py          # Entry point for Phase 1
│   ├── train_phase2.py          # Entry point for Phase 2
│   ├── train_phase3.py          # Entry point for Phase 3
│   ├── evaluate.py              # Run evaluation suite
│   └── demo.py                  # Interactive demo of inference-time learning
├── tests/
│   ├── test_adaptive_layer.py   # Unit tests for adaptive layer
│   ├── test_forward_pass.py     # Test full forward pass stability
│   └── test_learning.py         # Test that self-modification improves output
└── README.md
```

---

## Step 1: Implement the Adaptive Memory Layer

File: `nat/model/adaptive_layer.py`

This is the most important component. Get this right first.

### Architecture

```python
class AdaptiveMemoryLayer(nn.Module):
    """
    A self-modifying layer that learns from hidden states during inference.
    
    Has two types of parameters:
    - Slow parameters (θ): define the learning rule. Trained via meta-learning.
      These are standard nn.Module parameters that get gradients during training.
    - Fast weights (W): working memory that changes during inference.
      Stored as buffers, NOT parameters. Updated by the learning rule, not by
      the optimizer.
    
    The fast weights are low-rank: W = A @ B where A is (d_model, rank) 
    and B is (rank, d_model). This keeps them small (~100K-200K params).
    """
    
    def __init__(self, d_model: int, rank: int = 32, d_hidden: int = 256):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        
        # === SLOW PARAMETERS (θ) — trained, then frozen ===
        
        # Surprise network: reads hidden state prediction error, outputs surprise score
        # Input: d_model (the prediction error vector)
        # Output: 1 (scalar surprise score)
        self.surprise_net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()  # surprise in [0, 1]
        )
        
        # Learning rate network: surprise -> learning rate
        # Maps surprise magnitude to how fast W should update
        self.lr_net = nn.Sequential(
            nn.Linear(1, d_hidden // 4),
            nn.GELU(),
            nn.Linear(d_hidden // 4, 1),
            nn.Softplus()  # lr > 0 always
        )
        
        # Write network: (h_t, surprise) -> weight update delta_W
        # This defines WHAT to store in fast weights
        # We use an outer-product structure for efficiency:
        # delta_W = write_key(h_t) outer_product write_value(h_t)
        self.write_key_net = nn.Sequential(
            nn.Linear(d_model + 1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, rank)
        )
        self.write_value_net = nn.Sequential(
            nn.Linear(d_model + 1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        
        # Read network: queries fast weights with current hidden state
        self.read_net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        
        # Gate network: decides how much to trust memory vs passthrough
        # Input: h_t concatenated with memory_output
        # Output: scalar gate in [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )
        
        # State predictor: predicts next hidden state for surprise computation
        self.state_predictor = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        
        # Layer norm for output stability
        self.layer_norm = nn.LayerNorm(d_model)
        
        # === FAST WEIGHTS (W) — NOT optimizer parameters ===
        # Low-rank: W = fast_A @ fast_B
        # Learned initial values (these ARE optimizer parameters)
        self.fast_A_init = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.fast_B_init = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        
        # Runtime fast weights (set by reset_fast_weights)
        self.fast_A = None  # shape: (batch, d_model, rank)
        self.fast_B = None  # shape: (batch, rank, d_model)
        
        # Previous hidden state for surprise computation
        self.prev_h = None
    
    def reset_fast_weights(self, batch_size: int = 1):
        """Reset fast weights to learned initial values. Call at start of session."""
        # .clone() so each call gets a fresh copy
        # No .detach() — we need gradients during training
        self.fast_A = self.fast_A_init.clone().unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        self.fast_B = self.fast_B_init.clone().unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        self.prev_h = None
    
    def partial_reset(self, alpha: float = 0.5):
        """
        Partial reset between sessions. 
        alpha=1.0 is full reset to init, alpha=0.0 is no reset.
        """
        init_A = self.fast_A_init.unsqueeze(0).expand_as(self.fast_A)
        init_B = self.fast_B_init.unsqueeze(0).expand_as(self.fast_B)
        self.fast_A = alpha * init_A + (1 - alpha) * self.fast_A.detach()
        self.fast_B = alpha * init_B + (1 - alpha) * self.fast_B.detach()
    
    def adapt(self, h_t: torch.Tensor):
        """
        The self-modification step. Called every N tokens.
        
        CRITICAL: Every operation here must be differentiable during training.
        Use self.fast_A = self.fast_A + delta (creates new tensor in graph)
        NOT self.fast_A += delta (in-place, breaks autograd)
        
        Args:
            h_t: hidden state tensor, shape (batch, d_model)
                 Typically the mean of the last N hidden states.
        """
        # Step 1: Compute surprise
        if self.prev_h is not None:
            predicted_h = self.state_predictor(self.prev_h)
            error = h_t - predicted_h
            surprise = self.surprise_net(error)  # (batch, 1)
        else:
            # First adaptation step: maximum surprise
            surprise = torch.ones(h_t.shape[0], 1, device=h_t.device, dtype=h_t.dtype)
        
        # Step 2: Compute learning rate from surprise
        lr = self.lr_net(surprise)  # (batch, 1)
        lr = lr.clamp(max=0.1)     # Stability clamp
        
        # Step 3: Compute weight update
        write_input = torch.cat([h_t, surprise], dim=-1)  # (batch, d_model + 1)
        write_key = self.write_key_net(write_input)        # (batch, rank)
        write_value = self.write_value_net(write_input)    # (batch, d_model)
        
        # Rank-1 update to fast_A via outer product: 
        # delta_A = lr * write_value (outer) write_key
        # Shape: (batch, d_model, 1) @ (batch, 1, rank) = (batch, d_model, rank)
        delta_A = lr.unsqueeze(-1) * torch.bmm(
            write_value.unsqueeze(-1),  # (batch, d_model, 1)
            write_key.unsqueeze(-2)     # (batch, 1, rank)
        )
        
        # NOT in-place: creates new tensor in computation graph
        self.fast_A = self.fast_A + delta_A
        
        # Step 4: Store current h for next surprise computation
        # During training: keep in graph for potential gradient flow
        # During inference: detach to save memory
        self.prev_h = h_t if self.training else h_t.detach()
        
        # Step 5: Stability — normalize fast weights if too large
        with torch.no_grad():
            norm = torch.norm(self.fast_A, dim=(1, 2), keepdim=True)
            scale = torch.clamp(10.0 / (norm + 1e-8), max=1.0)
        self.fast_A = self.fast_A * scale  # scale is detached, so this is safe
    
    def read(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Read from fast weights and produce gated output.
        Called for every token (not just every N tokens).
        
        Args:
            h_t: hidden state tensor, shape (batch, seq_len, d_model)
        
        Returns:
            modified hidden state, shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = h_t.shape
        
        # Query fast weights: W @ h_t = (fast_A @ fast_B) @ h_t
        h_transposed = h_t.transpose(1, 2)  # (batch, d_model, seq_len)
        memory_raw = torch.bmm(
            self.fast_A, 
            torch.bmm(self.fast_B, h_transposed)
        ).transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Process through read network
        memory_output = self.read_net(memory_raw)  # (batch, seq_len, d_model)
        
        # Gate: decide how much to trust memory
        gate_input = torch.cat([h_t, memory_output], dim=-1)
        gate = self.gate_net(gate_input)  # (batch, seq_len, 1)
        
        # Residual connection with gated memory
        output = h_t + gate * memory_output
        return self.layer_norm(output)
    
    def forward(self, h_t: torch.Tensor, do_adapt: bool = False):
        """
        Full forward pass.
        
        Args:
            h_t: hidden states, shape (batch, seq_len, d_model)
            do_adapt: whether to run the adaptation step this call
        
        Returns:
            modified hidden states, shape (batch, seq_len, d_model)
        """
        if do_adapt:
            adapt_signal = h_t.mean(dim=1)  # (batch, d_model)
            self.adapt(adapt_signal)
        
        return self.read(h_t)
```

### Critical Implementation Notes

1. **Fast weights need gradients during training** for BPTT through the adaptation chain, but should NOT be optimizer parameters. They are regular tensors that participate in the computation graph during training.

2. **No in-place operations on fast weights.** `self.fast_A = self.fast_A + delta_A` creates a new node in the graph. `self.fast_A += delta_A` is in-place and breaks autograd. This distinction is critical.

3. **Batch dimension on fast weights**: Each item in a batch has its own fast weights `(batch, d_model, rank)`. This allows different items to learn different things.

4. **Numerical stability**: Fast weights can grow unbounded. The norm clamping in `adapt()` is essential. Add gradient clipping during training.

---

## Step 2: Implement the Consolidation Layer

File: `nat/model/consolidation_layer.py`

```python
class ConsolidationLayer(nn.Module):
    """
    Slow-learning layer that accumulates knowledge via EMA.
    
    During forward pass: read-only (queries W_c with hidden states).
    Between sessions: W_c <- beta * W_c + (1 - beta) * avg(W_A, W_B)
    """
    
    def __init__(self, d_model: int, rank: int = 32, d_hidden: int = 256, beta: float = 0.999):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.beta = beta
        
        self.read_net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Consolidated weights — persistent across sessions
        self.register_buffer('W_c_A', torch.zeros(1, d_model, rank))
        self.register_buffer('W_c_B', torch.zeros(1, rank, d_model))
    
    @torch.no_grad()
    def consolidate(self, adaptive_layers: list):
        """EMA update from adaptive layers. Call after each session."""
        avg_A = torch.stack([l.fast_A.mean(dim=0) for l in adaptive_layers]).mean(dim=0, keepdim=True)
        avg_B = torch.stack([l.fast_B.mean(dim=0) for l in adaptive_layers]).mean(dim=0, keepdim=True)
        self.W_c_A = self.beta * self.W_c_A + (1 - self.beta) * avg_A
        self.W_c_B = self.beta * self.W_c_B + (1 - self.beta) * avg_B
    
    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """Read-only forward pass."""
        batch, seq_len, d_model = h_t.shape
        W_A = self.W_c_A.expand(batch, -1, -1)
        W_B = self.W_c_B.expand(batch, -1, -1)
        
        h_T = h_t.transpose(1, 2)
        memory_raw = torch.bmm(W_A, torch.bmm(W_B, h_T)).transpose(1, 2)
        memory_output = self.read_net(memory_raw)
        
        gate = self.gate_net(torch.cat([h_t, memory_output], dim=-1))
        output = h_t + gate * memory_output
        return self.layer_norm(output)
    
    def save_state(self, path: str):
        torch.save({'W_c_A': self.W_c_A, 'W_c_B': self.W_c_B}, path)
    
    def load_state(self, path: str):
        state = torch.load(path, weights_only=True)
        self.W_c_A.copy_(state['W_c_A'])
        self.W_c_B.copy_(state['W_c_B'])
```

---

## Step 3: Implement the Full NAT Model

File: `nat/model/nat_model.py`

This is the hardest implementation step. You need to intercept the base model's forward pass between layers to insert the adaptive and consolidation layers.

```python
class NATModel(nn.Module):
    """
    Nested Adaptive Transformer.
    
    Wraps a frozen pretrained transformer with adaptive memory layers
    and a consolidation layer inserted between specific layers.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Load and freeze base model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze ALL base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get model config
        model_config = self.base_model.config
        self.d_model = model_config.hidden_size
        self.num_layers = model_config.num_hidden_layers
        
        # Insertion points
        self.insert_A = self.num_layers // 3
        self.insert_B = (2 * self.num_layers) // 3
        self.insert_C = (5 * self.num_layers) // 6
        
        # Create adaptive and consolidation layers in float32
        self.adaptive_A = AdaptiveMemoryLayer(self.d_model, config.rank, config.d_hidden).float()
        self.adaptive_B = AdaptiveMemoryLayer(self.d_model, config.rank, config.d_hidden).float()
        self.consolidation = ConsolidationLayer(
            self.d_model, config.rank, config.d_hidden, config.beta
        ).float()
        
        self.adapt_every_n = config.adapt_every_n
        self.config = config
    
    def get_trainable_parameters(self):
        """Return only parameters that should be trained."""
        params = []
        params.extend(self.adaptive_A.parameters())
        params.extend(self.adaptive_B.parameters())
        params.extend(self.consolidation.parameters())
        return params
    
    def start_session(self, batch_size: int = 1):
        self.adaptive_A.reset_fast_weights(batch_size)
        self.adaptive_B.reset_fast_weights(batch_size)
        self._step_counter = 0
    
    def end_session(self):
        self.consolidation.consolidate([self.adaptive_A, self.adaptive_B])
        self.adaptive_A.partial_reset(self.config.session_reset_alpha)
        self.adaptive_B.partial_reset(self.config.session_reset_alpha)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass with adaptive layer intervention.
        
        IMPORTANT: We manually run each transformer layer so we can
        insert our adaptive layers between them. This means we need
        to replicate the base model's forward logic.
        
        The implementation below is for Qwen2.5 / Llama-style models.
        You may need to adapt for other architectures.
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize fast weights if not already done
        if self.adaptive_A.fast_A is None:
            self.start_session(batch_size)
        
        # Get the transformer internals
        # For Qwen2.5: self.base_model.model
        # For Llama: self.base_model.model
        transformer = self.base_model.model
        
        # Embedding
        hidden_states = transformer.embed_tokens(input_ids)
        
        # Determine if we should adapt this step
        self._step_counter += seq_len
        do_adapt = (self._step_counter % self.adapt_every_n) < seq_len
        
        # Process through each layer
        for i, layer in enumerate(transformer.layers):
            # Run frozen transformer layer
            # We need to handle the layer's expected inputs carefully
            # Most models expect: hidden_states, attention_mask, position_ids, ...
            layer_outputs = layer(
                hidden_states.to(layer.self_attn.q_proj.weight.dtype),
                attention_mask=attention_mask,
                position_ids=None,  # Will be auto-computed by most models
            )
            hidden_states = layer_outputs[0].float()  # Back to float32 for adaptive layers
            
            # Insert adaptive/consolidation layers
            if i == self.insert_A:
                hidden_states = self.adaptive_A(hidden_states, do_adapt=do_adapt)
            elif i == self.insert_B:
                hidden_states = self.adaptive_B(hidden_states, do_adapt=do_adapt)
            elif i == self.insert_C:
                hidden_states = self.consolidation(hidden_states)
        
        # Final norm
        hidden_states = transformer.norm(hidden_states.to(transformer.norm.weight.dtype))
        
        # LM head
        logits = self.base_model.lm_head(hidden_states).float()
        
        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"loss": loss, "logits": logits}
```

### Critical: Getting the Layer-by-Layer Forward Pass Right

The above code is a template. The actual implementation depends on your base model. Here's how to figure it out:

```python
# Step 1: Inspect the model structure
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
print(model)
# This shows you the exact module names and structure

# Step 2: Look at the model's forward() source code
import inspect
print(inspect.getsource(model.model.forward))
# This shows you exactly how the model processes layers,
# handles attention masks, position embeddings, etc.

# Step 3: Replicate that logic with your insertions
```

Common issues:
- **Position IDs**: Some models compute position IDs internally, others expect them as input
- **Attention masks**: Causal masks may be created inside the model or need to be passed in
- **RoPE / rotary embeddings**: Applied inside each layer or passed in from outside
- **Residual connections**: Some models have pre-norm vs post-norm — your adaptive layer insertion must respect this

**Spend time getting this right.** A wrong attention mask or missing position embedding will silently produce bad results.

---

## Step 4: Implement Phase 1 Training

File: `nat/training/phase1_meta_learn.py`

```python
"""
Phase 1: Meta-learn the learning rule theta.

What is trained: theta (slow parameters of adaptive layers + consolidation layer)
What is NOT trained: base model weights, fast weights W

The gradient flows through the chain of self-modification steps (BPTT):
  L_eval -> logits -> frozen layers -> adaptive read -> 
  fast weights W_K -> W_{K-1} -> ... -> W_0 -> theta
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

def train_phase1(model, config):
    # Only optimize adaptive + consolidation parameters
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_episodes
    )
    
    dataloader = build_phase1_dataloader(config)
    
    for episode_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(config.device)  # (batch, seq_len)
        batch_size, seq_len = input_ids.shape
        
        # Split: 75% adaptation, 25% evaluation
        adapt_len = int(seq_len * 0.75)
        
        # Reset fast weights
        model.start_session(batch_size)
        
        # === ADAPTATION PHASE ===
        # Process in chunks. Each chunk triggers one adaptation step.
        chunk_size = config.adapt_every_n
        
        for chunk_start in range(0, adapt_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, adapt_len)
            chunk_ids = input_ids[:, chunk_start:chunk_end]
            
            # Forward pass — adaptive layers self-modify
            # No loss computation during adaptation
            _ = model(chunk_ids)
            
            # Optional: truncated BPTT
            # Every T steps, detach fast weights to limit gradient chain
            if config.truncated_bptt > 0:
                if (chunk_start // chunk_size) % config.truncated_bptt == 0:
                    model.adaptive_A.fast_A = model.adaptive_A.fast_A.detach().requires_grad_(True)
                    model.adaptive_A.fast_B = model.adaptive_A.fast_B.detach().requires_grad_(True)
                    model.adaptive_B.fast_A = model.adaptive_B.fast_A.detach().requires_grad_(True)
                    model.adaptive_B.fast_B = model.adaptive_B.fast_B.detach().requires_grad_(True)
        
        # === EVALUATION PHASE ===
        # Measure how good the adapted fast weights are
        eval_ids = input_ids[:, adapt_len:]
        eval_labels = eval_ids.clone()
        
        output = model(eval_ids, labels=eval_labels)
        loss = output["loss"]
        
        # === ALSO COMPUTE BASELINE ===
        # What's the loss WITHOUT adaptation? (for comparison only)
        with torch.no_grad():
            model_copy_A = model.adaptive_A.fast_A.clone()
            model_copy_B = model.adaptive_B.fast_B.clone()
            model.start_session(batch_size)  # reset to init
            baseline_output = model(eval_ids, labels=eval_labels)
            baseline_loss = baseline_output["loss"]
            # Restore adapted weights
            model.adaptive_A.fast_A = model_copy_A
            model.adaptive_B.fast_B = model_copy_B
        
        # === BACKPROP ===
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if episode_idx % 50 == 0:
            adaptation_benefit = baseline_loss.item() - loss.item()
            print(f"Episode {episode_idx}: "
                  f"loss={loss.item():.4f} "
                  f"baseline={baseline_loss.item():.4f} "
                  f"benefit={adaptation_benefit:.4f}")
            wandb.log({
                "loss": loss.item(),
                "baseline_loss": baseline_loss.item(),
                "adaptation_benefit": adaptation_benefit,
                "lr": scheduler.get_last_lr()[0]
            })
        
        if episode_idx >= config.num_episodes:
            break
    
    # Save trained theta
    torch.save({
        "adaptive_A": model.adaptive_A.state_dict(),
        "adaptive_B": model.adaptive_B.state_dict(),
        "consolidation": model.consolidation.state_dict(),
    }, config.save_path)


def build_phase1_dataloader(config):
    """
    Build diverse training data. Each item is a sequence of 2048 tokens
    from a diverse mix of tasks.
    
    Use HuggingFace datasets:
    - gsm8k (math)
    - openai_humaneval (code)
    - allenai/ai2_arc (reasoning)
    - squad (reading comprehension)
    - c4 (general text)
    """
    from datasets import load_dataset, concatenate_datasets
    
    # Load multiple datasets and interleave
    datasets_list = []
    
    # Example: load C4 for general text
    c4 = load_dataset("c4", "en", split="train", streaming=True)
    # ... tokenize and chunk into seq_len sequences ...
    
    # Example: load GSM8K for math
    gsm8k = load_dataset("gsm8k", "main", split="train")
    # ... format as "Question: ... Answer: ..." and tokenize ...
    
    # Interleave datasets
    # ... implementation depends on your specific setup ...
    
    pass  # Implement based on your data needs
```

### The BPTT Implementation Challenge

The adaptation chain must remain in the computation graph during training. Here's a checklist:

- [ ] `fast_A` and `fast_B` are created with gradients enabled (via `fast_A_init.clone()` without `.detach()`)
- [ ] The `adapt()` method uses `self.fast_A = self.fast_A + delta` (NOT `+=`)
- [ ] The norm clamping uses `torch.no_grad()` for the scale computation but applies the scale via multiplication (which IS in the graph)
- [ ] No stray `.detach()` calls breaking the chain
- [ ] The loss on the evaluation portion has a path back through the adapted fast weights

**Test**: After one training step, verify that `model.adaptive_A.fast_A_init.grad` is not None. If it's None, the gradient chain is broken.

---

## Step 5: Implement Phase 2 Training

File: `nat/training/phase2_episodic.py`

Same loop as Phase 1, but with structured episodic data.

```python
def build_episodic_data(task: str, num_problems: int = 8):
    """
    Build an episode: a sequence of related problems of increasing difficulty.
    
    Format:
    [Problem 1]\n[Solution 1]\n\n[Problem 2]\n[Solution 2]\n\n...
    
    The model should improve from problem 1 to problem N because the
    adaptive layers learn from the hidden states of earlier problems.
    """
    if task == "math":
        # Sample GSM8K problems, order by difficulty (solution length as proxy)
        problems = sample_gsm8k(num_problems, sort_by_difficulty=True)
    elif task == "code":
        # Sample related coding problems
        problems = sample_humaneval(num_problems, same_category=True)
    # ... etc
    
    # Concatenate with clear separators
    text = "\n\n".join([
        f"Problem {i+1}: {p['question']}\nSolution: {p['answer']}"
        for i, p in enumerate(problems)
    ])
    
    return tokenize(text)


def compute_episodic_loss(model, input_ids, problem_spans, config):
    """
    Compute loss with improvement bonus.
    
    problem_spans: list of (solution_start, solution_end) token indices
    """
    output = model(input_ids, labels=input_ids)
    logits = output["logits"]
    
    # Per-problem loss
    problem_losses = []
    for sol_start, sol_end in problem_spans:
        sol_logits = logits[:, sol_start-1:sol_end-1, :]
        sol_labels = input_ids[:, sol_start:sol_end]
        prob_loss = F.cross_entropy(
            sol_logits.reshape(-1, sol_logits.size(-1)),
            sol_labels.reshape(-1)
        )
        problem_losses.append(prob_loss)
    
    # Average loss
    base_loss = torch.stack(problem_losses).mean()
    
    # Improvement bonus: reward decreasing loss across problems
    improvement = torch.tensor(0.0, device=base_loss.device)
    for i in range(1, len(problem_losses)):
        improvement = improvement + torch.relu(problem_losses[i-1] - problem_losses[i])
    improvement = improvement / max(1, len(problem_losses) - 1)
    
    total_loss = base_loss - config.improvement_weight * improvement
    
    return total_loss, [l.item() for l in problem_losses]
```

---

## Step 6: Phase 3 Training (Consolidation)

File: `nat/training/phase3_consolidation.py`

```python
def train_phase3(model, config):
    """
    Train consolidation dynamics: beta (EMA rate) and alpha (reset rate).
    
    Freeze adaptive layer theta. Only train consolidation parameters.
    """
    # Freeze adaptive layers
    for param in model.adaptive_A.parameters():
        param.requires_grad = False
    for param in model.adaptive_B.parameters():
        param.requires_grad = False
    
    # Make beta and alpha trainable
    beta_param = nn.Parameter(torch.tensor(0.999))
    alpha_param = nn.Parameter(torch.tensor(0.5))
    
    optimizer = torch.optim.Adam([
        *model.consolidation.parameters(),
        beta_param,
        alpha_param
    ], lr=1e-4)
    
    domains = ["math", "code", "reasoning", "text", "science"]
    
    for run_idx in range(config.num_runs):
        # Build domain sequence: D1*20 -> D2*20 -> D1*5 (forgetting test)
        sequence = build_domain_sequence(domains, config)
        
        # Reset consolidation
        model.consolidation.W_c_A.zero_()
        model.consolidation.W_c_B.zero_()
        
        all_results = {d: [] for d in domains}
        
        for domain in sequence:
            # Run episode
            model.start_session(batch_size=1)
            episode = sample_episode(domain)
            output = model(episode["input_ids"], labels=episode["labels"])
            loss = output["loss"]
            all_results[domain].append(loss.item())
            
            # Consolidate with current beta
            model.consolidation.beta = torch.sigmoid(beta_param).item()  # keep in (0,1)
            model.end_session()
        
        # Compute training signal
        # ... reward cross-session improvement, penalize forgetting ...
        # ... backprop to update consolidation parameters, beta, alpha ...
```

---

## Step 7: Evaluation Suite

File: `nat/training/eval.py`

Implement three core evaluations:

1. **Within-session learning**: Does loss decrease from problem 1 to problem 10?
2. **Cross-session learning**: Does session 20 start better than session 1?
3. **Forgetting test**: After learning domain B, is domain A preserved?

See the architecture document for detailed evaluation protocols.

---

## Step 8: Running on Apple Silicon

```python
# For M1/M2/M3/M4 Macs:
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Key settings for Apple Silicon:
config.batch_size = 1          # Memory limited
config.seq_len = 1024          # Shorter sequences
config.base_dtype = torch.bfloat16  # Supported on MPS
config.adapt_every_n = 64     # Less frequent adaptation = less memory

# Expected timeline on M1 16GB:
# Phase 1: ~3-5 days
# Phase 2: ~3-5 days
# Phase 3: ~1-2 weeks
# Total: ~3-4 weeks
```

---

## Hyperparameter Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| base_model_name | Qwen/Qwen2.5-1.5B | Frozen |
| rank | 32 | Fast weight rank |
| d_hidden | 256 | Hidden dim for theta networks |
| adapt_every_n | 32 | Tokens between adaptation steps |
| lr (Phase 1-2) | 3e-4 | Adam on theta |
| lr (Phase 3) | 1e-4 | Adam on consolidation |
| batch_size | 4 (A100) / 1 (M1) | |
| seq_len | 2048 (A100) / 1024 (M1) | |
| beta | 0.999 | EMA consolidation rate |
| alpha | 0.5 | Session reset rate |
| grad_clip | 1.0 | Max gradient norm |
| lr_clamp | 0.1 | Max adaptive learning rate |
| fast_weight_max_norm | 10.0 | Stability constraint |
| truncated_bptt | 16 | 0 = full BPTT |
| improvement_weight | 0.1 | Phase 2 improvement bonus |
| num_episodes_p1 | 50000 | Phase 1 episodes |
| num_episodes_p2 | 30000 | Phase 2 episodes |
| num_runs_p3 | 100 | Phase 3 consolidation runs |

---

## Milestones and Success Criteria

### Milestone 1: Stable Forward Pass (Week 1-2)
- [ ] Forward pass produces valid logits (no NaN/Inf)
- [ ] Adaptive layer self-modification runs without crashing
- [ ] Output CHANGES when adaptive layer is active (verify it's actually doing something)
- [ ] Fast weights actually change during forward pass (print norms before/after)

### Milestone 2: Meta-Learning Works (Week 3-4)
- [ ] Gradients flow through the adaptation chain (theta.grad is not None)
- [ ] Training loss decreases over episodes
- [ ] **KEY METRIC**: eval loss WITH adaptation < eval loss WITHOUT adaptation
- [ ] The adaptation benefit increases over training (theta is improving)

### Milestone 3: Episodic Improvement (Week 5-6)
- [ ] **KEY METRIC**: loss on problem 8 < loss on problem 1 within episodes
- [ ] The improvement is not explained by attention alone
  (test: mask previous solutions, does improvement persist via fast weights?)

### Milestone 4: Cross-Session Learning (Week 7-8)
- [ ] **KEY METRIC**: session 20 outperforms session 1 on same domain
- [ ] **KEY METRIC**: after 10 domain switches, original domain < 5% degraded

### Milestone 5: Full Evaluation (Week 9-10)
- [ ] Learning curves plotted for within-session and cross-session
- [ ] Forgetting curves plotted across domain switches
- [ ] Comparison to baselines: frozen model, LoRA fine-tuning
- [ ] Results documented

---

## Common Pitfalls and Debugging

1. **NaN in fast weights**: Learning rate too high. Reduce `lr_clamp` from 0.1 to 0.01. Check that norm clamping is working.

2. **No gradient to theta**: The computation graph is broken. Print `fast_A.grad_fn` after each adaptation step — it should show a chain of operations, not None. Look for accidental `.detach()` calls.

3. **Adaptation makes things worse**: The learning rule hasn't learned yet (early training), or the surprise signal is uninformative. Try a simpler surprise signal first (L2 norm of hidden state change instead of the learned surprise network).

4. **Gate always 0**: The adaptive layer learned to shut itself off. Initialize gate bias to -1.0 so initial gate is ~0.27 (slightly open). Add a small penalty for gate being exactly 0.

5. **Gate always 1**: The adaptive layer is overriding the frozen model entirely. Add regularization: `loss += 0.01 * gate.mean()`.

6. **Base model forward pass breaks**: You're not replicating the base model's internal logic correctly. Compare outputs: `your_forward(x)` vs `base_model(x)` with adaptive layers disabled. They should be identical. If not, debug the attention mask / position embedding handling.

7. **OOM during BPTT**: Too many adaptation steps in the graph. Use truncated BPTT (detach every 8-16 steps) or gradient checkpointing. Reduce batch size to 1.

8. **Training is too slow**: The frozen model's forward pass dominates compute. Make sure you're using bf16/fp16 for the frozen layers. Make sure frozen parameters have `requires_grad=False` (saves memory for activations that don't need gradient tracking).