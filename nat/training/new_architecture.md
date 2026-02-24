# NAT Migration: Qwen 2.5 1.5B → Qwen 3 4B
# Final Prompt for Coding Agent

You have a working NAT implementation targeting Qwen 2.5 1.5B.
Modify it to use Qwen 3 4B on an A100 80GB. Do NOT rewrite from scratch.

---

## Part 1: Model and Architecture Changes

### 1.1 Swap the base model

```
OLD: Qwen/Qwen2.5-1.5B
NEW: Qwen/Qwen3-4B
```

Requires `transformers >= 4.51.0`:
```bash
pip install --upgrade transformers
```

### 1.2 Auto-detect dimensions

Do not hardcode dimensions anywhere. Read from the model config:

```python
d_model = base_model.config.hidden_size         # will be 2560
num_layers = base_model.config.num_hidden_layers # will be 36
```

Update all dimension-dependent code:
- AdaptiveMemoryLayer: all Linear layers using d_model
- ConsolidationLayer: all Linear layers using d_model
- fast_A_init shape: (d_model, rank)
- fast_B_init shape: (rank, d_model)
- Any config/yaml files

### 1.3 Update insertion points

Compute dynamically from num_layers:

```python
insert_A = num_layers // 3           # 12 for 36 layers
insert_B = (2 * num_layers) // 3     # 24
insert_C = (5 * num_layers) // 6     # 30
```

### 1.4 Increase rank and hidden dim

```python
rank = 48       # was 32
d_hidden = 384  # was 256
```

Trainable params go from ~13M to ~25M. Still < 1% of base model.

### 1.5 Switch forward pass to hooks

Replace the manual layer-by-layer loop with PyTorch forward hooks.
Qwen3 has different internals than Qwen 2.5 (QK-Norm, different RoPE,
different GQA). Hooks avoid replicating those internals.

**First, inspect what each layer returns:**

```python
import torch, inspect
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Check layer output format
dummy = tokenizer("test", return_tensors="pt")
with torch.no_grad():
    embedded = model.model.embed_tokens(dummy["input_ids"])
    output = model.model.layers[0](embedded)
    print(f"Type: {type(output)}")
    print(f"Length: {len(output)}")
    for i, o in enumerate(output):
        if isinstance(o, torch.Tensor):
            print(f"  [{i}] Tensor shape: {o.shape}")
        else:
            print(f"  [{i}] {type(o)}")

# Check layer forward signature
print(inspect.signature(model.model.layers[0].forward))
```

**Then register hooks:**

```python
def _register_hooks(self):
    layers = self.base_model.model.layers

    def make_hook(adaptive_layer):
        def hook(module, input, output):
            # Modify output[0] (hidden states), preserve rest of tuple
            hidden = output[0].float()
            modified = adaptive_layer(hidden, do_adapt=self._do_adapt, config=self.config)
            return (modified.to(output[0].dtype),) + output[1:]
        return hook

    def consolidation_hook(module, input, output):
        hidden = output[0].float()
        modified = self.consolidation(hidden)
        return (modified.to(output[0].dtype),) + output[1:]

    layers[self.insert_A].register_forward_hook(make_hook(self.adaptive_A))
    layers[self.insert_B].register_forward_hook(make_hook(self.adaptive_B))
    layers[self.insert_C].register_forward_hook(consolidation_hook)
```

**The forward method becomes:**
```python
def forward(self, input_ids, attention_mask=None, labels=None):
    return self.base_model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        labels=labels
    )
```

**CRITICAL: The hook must return the EXACT same tuple format as the
original layer output. Check the inspection output above to confirm.**

### 1.6 Verify gradient flow through hooks

Run this IMMEDIATELY after implementing hooks:

```python
model.start_session(batch_size=1)

# Run adaptation chunks
for i in range(4):
    chunk = dummy_ids[:, i*64:(i+1)*64].to(device)
    _ = model(chunk)

# Run eval with loss
eval_ids = dummy_ids[:, 256:].to(device)
output = model(eval_ids, labels=eval_ids)
output.loss.backward()

# ALL must be True:
assert model.adaptive_A.fast_A_init.grad is not None, "No gradient to fast_A_init!"
assert list(model.adaptive_A.surprise_net.parameters())[0].grad is not None, "No gradient to surprise_net!"
assert list(model.adaptive_A.write_key_net.parameters())[0].grad is not None, "No gradient to write_key_net!"
```

If any assertion fails, gradients don't flow through hooks. Fall back to
the manual layer loop approach, updated for Qwen3 internals using the
signature and structure you inspected above.

### 1.7 Update hyperparameters for 80GB

```yaml
batch_size: 4
seq_len: 2048
truncated_bptt: 16
adapt_every_n: 64
lr: 2e-4
lr_clamp: 0.05
fast_weight_max_norm: 8.0
grad_clip: 1.0
gradient_checkpointing: true
```

---

## Part 2: Training Phase Restructure

The old three-phase structure is eliminated. The new structure has two phases.

### Delete old Phase 1 (Wikipedia/C4)

Remove entirely. Delete the training script, data loader, and config.
Wikipedia text has no reasoning traces. Training the learning rule on
flat text wastes time and miscalibrates the surprise network.

### Rename old Phase 2 → new Phase 1 (Meta-Learning on AMPS)

### Rename old Phase 3 → new Phase 2 (Consolidation across domains)

Update file names, script entry points, config references, and README.

---

## Part 3: Phase 1 — Meta-Learn θ on All Domains

### 3.1 Datasets

Use all five domains from the start. More diversity means θ learns a
more general learning rule — one that works for math, code, logic,
reading, and science, not just math.

| Domain | Dataset | Grouping | Reasoning Traces |
|--------|---------|----------|-----------------|
| Math | AMPS Khan Academy | Exercise type (693 types) | Built-in (LaTeX solutions) |
| Math (hard) | MATH (Hendrycks) | Subject + difficulty level | Built-in (LaTeX solutions) |
| Code | CodeForces-CoTs (open-r1) | Algorithm tag via TACO | Built-in (DeepSeek-R1 CoT) |
| Logic | AR-LSAT | Shared scenario (5-7 Qs each) | Generate via reasoning model |
| Reading | DROP | Shared passage (~14 Qs each) | Generate via reasoning model |
| Science | ScienceQA | Skill category (379 skills) | Built-in (explanations) |

### 3.2 Generating missing reasoning traces

AR-LSAT and DROP need CoT traces generated before training.
This is a one-time preprocessing step, NOT done during training.

```python
def generate_cot_for_dataset(dataset, reasoning_model, tokenizer):
    """
    Generate chain-of-thought reasoning traces for datasets that lack them.
    Run ONCE. Save results to disk.
    
    Uses a reasoning model (e.g. the Qwen3-4B base model itself, or a
    larger model if available) to generate step-by-step solutions.
    Only keeps traces where the generated final answer matches ground truth.
    """
    enriched = []
    
    for problem in dataset:
        question = problem["question"]
        ground_truth = problem["answer"]
        
        # Generate reasoning trace
        prompt = f"Solve this step by step.\n\nQuestion: {question}\n\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt").to(reasoning_model.device)
        
        with torch.no_grad():
            output = reasoning_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.95
            )
        
        generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], 
                                     skip_special_tokens=True)
        
        # Extract final answer from generated text and compare to ground truth
        generated_answer = extract_final_answer(generated)
        
        if answers_match(generated_answer, ground_truth):
            problem["solution"] = generated.strip()
            enriched.append(problem)
        # else: discard — bad trace
    
    return enriched

# Run once for each dataset needing traces:
# ar_lsat_enriched = generate_cot_for_dataset(ar_lsat, model, tokenizer)
# drop_enriched = generate_cot_for_dataset(drop, model, tokenizer)
# Save to disk: json.dump(enriched, open("data/ar_lsat_with_cot.json", "w"))
```

### 3.3 Episode construction

Each episode is a group of 5-8 related problems from the same group.
The grouping logic differs by domain, but the output format is the same:
plain text with problems and step-by-step solutions.

```python
def build_episode(problems, tokenizer, num_problems=8):
    """
    Build one training episode from a list of related problems.
    Works for any domain — problems must have 'problem' and 'solution' fields.
    """
    selected = random.sample(problems, min(num_problems, len(problems)))
    
    text = ""
    for i, prob in enumerate(selected):
        text += f"Problem {i+1}: {prob['problem']}\n"
        text += f"Solution: {prob['solution']}\n\n"
    
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=False
    )
    return tokens
```

**Do NOT add `<think>` tags, chat templates, or special formatting.**
The solutions are already step-by-step reasoning traces. The model's
hidden states while reading these traces are the learning signal.

### 3.4 Data loader

```python
class MultiDomainEpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, domain_data, tokenizer, problems_per_episode=8):
        """
        Args:
            domain_data: dict of {domain_name: {group_key: [problems]}}
                Example: {
                    "math": {"addition": [...], "fractions": [...], ...},
                    "code": {"dp": [...], "greedy": [...], ...},
                    "logic": {"scenario_1": [...], "scenario_2": [...], ...},
                    "reading": {"passage_1": [...], "passage_2": [...], ...},
                    "science": {"skill_1": [...], "skill_2": [...], ...},
                }
        """
        self.tokenizer = tokenizer
        self.problems_per_episode = problems_per_episode
        
        # Flatten to list of (domain, group_key, problems)
        self.groups = []
        for domain, groups in domain_data.items():
            for group_key, problems in groups.items():
                if len(problems) >= problems_per_episode:
                    self.groups.append({
                        "domain": domain,
                        "group_key": group_key,
                        "problems": problems
                    })
        
        print(f"Loaded {len(self.groups)} groups across {len(domain_data)} domains")
    
    def __len__(self):
        return 50000  # num_episodes, configurable
    
    def __getitem__(self, idx):
        group = random.choice(self.groups)
        tokens = build_episode(group["problems"], self.tokenizer, 
                              self.problems_per_episode)
        tokens["domain"] = group["domain"]
        return tokens
```

### 3.4 Training loop

Same structure as the old Phase 2 (episodic training). Each episode:

1. Reset fast weights (start_session)
2. Process adaptation portion (first 75%) in chunks
   - Adaptive layers self-modify at each chunk
   - Truncated BPTT every 16 chunks
3. Process evaluation portion (last 25%) with loss
4. Compute baseline loss without adaptation (for comparison)
5. Backprop loss to update θ
6. Log adaptation_benefit = baseline_loss - adapted_loss

```python
def train_phase1(model, config):
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_episodes
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    dataset = AMPSEpisodeDataset(amps_data, tokenizer, config.problems_per_episode)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    for episode_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(config.device)
        batch_size, seq_len = input_ids.shape
        
        adapt_len = int(seq_len * 0.75)
        
        # Reset fast weights
        model.start_session(batch_size)
        model._do_adapt = True
        
        # === ADAPTATION ===
        chunk_size = config.adapt_every_n
        bptt_counter = 0
        
        for chunk_start in range(0, adapt_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, adapt_len)
            chunk_ids = input_ids[:, chunk_start:chunk_end]
            _ = model(chunk_ids)
            
            bptt_counter += 1
            if config.truncated_bptt > 0 and bptt_counter % config.truncated_bptt == 0:
                model.adaptive_A.fast_A = model.adaptive_A.fast_A.detach().requires_grad_(True)
                model.adaptive_A.fast_B = model.adaptive_A.fast_B.detach().requires_grad_(True)
                model.adaptive_B.fast_A = model.adaptive_B.fast_A.detach().requires_grad_(True)
                model.adaptive_B.fast_B = model.adaptive_B.fast_B.detach().requires_grad_(True)
        
        # === EVALUATION ===
        model._do_adapt = False
        eval_ids = input_ids[:, adapt_len:]
        output = model(eval_ids, labels=eval_ids)
        loss = output.loss
        
        # === BASELINE (no gradient) ===
        with torch.no_grad():
            saved = {
                'A_A': model.adaptive_A.fast_A.clone(),
                'A_B': model.adaptive_A.fast_B.clone(),
                'B_A': model.adaptive_B.fast_A.clone(),
                'B_B': model.adaptive_B.fast_B.clone(),
            }
            model.start_session(batch_size)
            baseline_loss = model(eval_ids, labels=eval_ids).loss
            model.adaptive_A.fast_A = saved['A_A']
            model.adaptive_A.fast_B = saved['A_B']
            model.adaptive_B.fast_A = saved['B_A']
            model.adaptive_B.fast_B = saved['B_B']
        
        # === BACKPROP ===
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        
        # === LOGGING ===
        if episode_idx % 50 == 0:
            benefit = baseline_loss.item() - loss.item()
            print(f"Ep {episode_idx}: loss={loss.item():.4f} "
                  f"baseline={baseline_loss.item():.4f} benefit={benefit:.4f}")
            wandb.log({
                "episode": episode_idx,
                "loss": loss.item(),
                "baseline_loss": baseline_loss.item(),
                "adaptation_benefit": benefit,
            })
        
        if episode_idx % 5000 == 0 and episode_idx > 0:
            save_checkpoint(model, optimizer, episode_idx, config)
        
        del loss, output
        torch.cuda.empty_cache()
        
        if episode_idx >= config.num_episodes:
            break
    
    save_checkpoint(model, optimizer, config.num_episodes, config)
```

### 3.5 Phase 1 config

```yaml
# configs/phase1.yaml
base_model_name: Qwen/Qwen3-4B
datasets:
  - name: amps_khan_academy
    grouping: exercise_type
  - name: math_hendrycks
    grouping: subject_and_level
  - name: codeforces_cots
    grouping: algorithm_tag
  - name: ar_lsat
    grouping: shared_scenario
    cot_generated: true
  - name: drop
    grouping: shared_passage
    cot_generated: true
  - name: scienceqa
    grouping: skill_category
problems_per_episode: 8
num_episodes: 50000
batch_size: 4
seq_len: 2048
rank: 48
d_hidden: 384
adapt_every_n: 64
truncated_bptt: 16
lr: 2e-4
lr_clamp: 0.05
fast_weight_max_norm: 8.0
grad_clip: 1.0
gradient_checkpointing: true
```

### 3.6 Phase 1 success criteria

- adaptation_benefit > 0 consistently after 5000 episodes
- benefit is positive across ALL domains (not just math)
- benefit increases over training (θ is improving)
- benefit plateaus at a stable positive value by 30000-50000 episodes
- fast weight norms are stable (not growing unbounded)
- no NaN anywhere
- memory usage stable over episodes (no leak)

---

## Part 4: Phase 2 — Consolidation Across Domains

Phase 2 tests whether knowledge persists across sessions and domain switches.
θ is frozen from Phase 1. Only consolidation parameters are trained.

Phase 1 trained θ on interleaved episodes from all domains — but each
episode was independent. The fast weights reset between episodes.
Phase 2 tests what happens when fast weights are consolidated into
persistent slow weights and carried across multiple sessions.

### 4.1 Datasets

Same five domains as Phase 1. Already preprocessed with reasoning traces.

### 4.2 Episode construction per domain

Same structure as Phase 1 — group related problems, concatenate with
solutions, tokenize. Each domain has its own grouping logic:

```python
def build_episode_for_domain(domain, data, tokenizer):
    if domain == "math":
        # Group by exercise type (same as Phase 1)
        return build_amps_episode(data, tokenizer)
    elif domain == "code":
        # Group by algorithm tag
        return build_code_episode(data, tokenizer)
    elif domain == "logic":
        # Group by shared scenario (AR-LSAT gives you this naturally)
        return build_logic_episode(data, tokenizer)
    elif domain == "reading":
        # Group by shared passage (DROP gives you this naturally)
        return build_reading_episode(data, tokenizer)
    elif domain == "science":
        # Group by skill category
        return build_science_episode(data, tokenizer)
```

### 4.3 Training loop

```python
def train_phase2(model, config):
    """
    Freeze θ. Train consolidation parameters (beta, alpha, consolidation networks).
    
    Structure: for each run, do a domain sequence like:
    D1 × 20 sessions → D2 × 20 sessions → D1 × 5 sessions (forgetting test)
    """
    # Freeze adaptive layer θ
    for param in model.adaptive_A.parameters():
        param.requires_grad = False
    for param in model.adaptive_B.parameters():
        param.requires_grad = False
    
    # Trainable: consolidation parameters + beta + alpha
    beta_logit = nn.Parameter(torch.tensor(6.9))   # sigmoid(6.9) ≈ 0.999
    alpha_logit = nn.Parameter(torch.tensor(0.0))   # sigmoid(0.0) = 0.5
    
    optimizer = torch.optim.Adam([
        *model.consolidation.parameters(),
        beta_logit,
        alpha_logit
    ], lr=1e-4)
    
    domains = ["math", "code", "logic", "reading", "science"]
    
    for run_idx in range(config.num_consolidation_runs):
        # Pick two domains
        d1, d2 = random.sample(domains, 2)
        sequence = [d1] * 20 + [d2] * 20 + [d1] * 5
        
        # Reset consolidation state
        model.consolidation.W_c_A.data.zero_()
        model.consolidation.W_c_B.data.zero_()
        
        results = {d1: [], d2: []}
        
        for domain in sequence:
            model.consolidation.beta = torch.sigmoid(beta_logit).item()
            model.start_session(batch_size=1)
            model._do_adapt = True
            
            episode = build_episode_for_domain(domain, datasets[domain], tokenizer)
            input_ids = episode["input_ids"].to(config.device)
            seq_len = input_ids.shape[1]
            adapt_len = int(seq_len * 0.75)
            
            # Adaptation
            chunk_size = config.adapt_every_n
            for chunk_start in range(0, adapt_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, adapt_len)
                _ = model(input_ids[:, chunk_start:chunk_end])
            
            # Eval
            model._do_adapt = False
            eval_ids = input_ids[:, adapt_len:]
            output = model(eval_ids, labels=eval_ids)
            results[domain].append(output.loss.item())
            
            # Consolidate
            model.end_session()
        
        # Measure forgetting
        d1_before = sum(results[d1][:20]) / 20
        d1_after = sum(results[d1][20:]) / max(1, len(results[d1][20:]))
        forgetting = max(0, d1_after - d1_before)
        
        # Cross-session improvement
        d1_first5 = sum(results[d1][:5]) / 5
        d1_last5_before_switch = sum(results[d1][15:20]) / 5
        improvement = max(0, d1_first5 - d1_last5_before_switch)
        
        # Loss: penalize forgetting, reward improvement
        run_loss = forgetting - config.improvement_weight * improvement
        
        optimizer.zero_grad()
        # Note: run_loss is a scalar from .item() calls, not a tensor.
        # You need to make this differentiable. One approach: accumulate
        # the actual tensor losses during the sequence instead of .item().
        # This is left as an implementation detail — the key is that
        # gradients flow to beta_logit and consolidation parameters.
        
        if run_idx % 10 == 0:
            print(f"Run {run_idx}: forgetting={forgetting:.4f} "
                  f"improvement={improvement:.4f} "
                  f"beta={torch.sigmoid(beta_logit).item():.6f}")

### 4.4 Phase 2 config

```yaml
# configs/phase2.yaml
num_consolidation_runs: 100
sessions_per_domain: 20
forgetting_test_sessions: 5
improvement_weight: 0.5
forgetting_penalty: 2.0
consolidation_lr: 1e-4
```

### 4.5 Phase 2 success criteria

- Cross-session improvement: session 20 loss < session 1 loss (same domain)
- Low forgetting: after 20 sessions in domain B, domain A < 10% degraded
- Beta converges to a stable value (likely 0.995-0.9999)
- Consolidation state norms are stable

---

## Part 5: What NOT to Change

- Adaptive layer architecture (surprise, lr, write, read, gate networks)
- Consolidation layer EMA logic
- Fast weight update rule (self.fast_A = self.fast_A + delta_A, NOT +=)
- Norm clamping logic
- Session management (start_session, end_session)
- BPTT structure (adaptation chunks → eval → backward)
- adaptation_benefit metric (baseline_loss - adapted_loss)

---

## Part 6: Warnings

### Gradient flow through hooks
After implementing, verify gradients reach θ. If they don't, fall back
to the manual layer loop updated for Qwen3 internals.

### Memory leak
Run 20 episodes, print GPU memory each time:
```python
for ep in range(20):
    # ... run episode ...
    print(f"Ep {ep}: {torch.cuda.memory_allocated()/1e9:.1f} GB")
```
Must NOT increase. If it does: missing .detach() on prev_h or
fast weights accumulating in the graph across episodes.

### Surprise saturation
Monitor surprise values. Should range 0.1-0.8 with variation.
If saturated at 1.0: reduce lr_clamp to 0.01, increase d_hidden
in state_predictor network.

### Gate saturation
Gates should average 0.1-0.5. If always near 0: adaptive layers
learned to shut off (check gate bias init is -1.0). If always near 1:
adaptive layers dominating (add regularization: loss += 0.01 * gate.mean()).

### NaN in fast weights
Reduce lr_clamp. Check norm clamping is working. Print fast_A norm
every 100 episodes to catch drift early.

### Hook output format
The hook MUST return the exact same tuple structure as the original
layer output. Inspect the layer output BEFORE writing hooks. If
the format is wrong you'll get cryptic errors deep in the model.

### LaTeX in AMPS
AMPS solutions use LaTeX formatting (\frac{}, \boxed{}, etc).
Make sure the tokenizer handles these correctly. Qwen3's tokenizer
should handle LaTeX fine, but verify on a few examples:
```python
text = r"Solution: \frac{d}{dx}(3x^2) = 6x"
tokens = tokenizer(text)
decoded = tokenizer.decode(tokens["input_ids"])
print(decoded)  # Should round-trip cleanly
```

---

## Summary Checklist

Architecture:
- [ ] Model name → Qwen/Qwen3-4B
- [ ] transformers >= 4.51.0
- [ ] d_model auto-detected (2560)
- [ ] num_layers auto-detected (36)
- [ ] Insertion points computed dynamically (12, 24, 30)
- [ ] rank → 48, d_hidden → 384
- [ ] Forward pass uses hooks
- [ ] Hook output format verified
- [ ] Gradient flow verified through hooks
- [ ] Hyperparameters updated for 80GB

Training restructure:
- [ ] Old Phase 1 (Wikipedia) deleted
- [ ] Old Phase 2 → new Phase 1 (AMPS)
- [ ] Old Phase 3 → new Phase 2 (multi-domain consolidation)

Phase 1 data:
- [ ] AMPS Khan Academy loaded, grouped by exercise type
- [ ] MATH (Hendrycks) loaded, grouped by subject + level
- [ ] CodeForces-CoTs loaded, grouped by algorithm tag
- [ ] AR-LSAT loaded, CoT generated, grouped by scenario
- [ ] DROP loaded, CoT generated, grouped by passage
- [ ] ScienceQA loaded, grouped by skill category
- [ ] All datasets normalized to {problem, solution} format
- [ ] Episodes: 5-8 related problems per episode
- [ ] Format: plain text, problem + step-by-step solution
- [ ] No chat template, no <think> tags
- [ ] Loss on full sequence (standard language modeling)

Phase 2 data:
- [ ] Same datasets as Phase 1, already preprocessed

Verification:
- [ ] adaptation_benefit > 0 after 5000 episodes
- [ ] Memory stable over 20 episodes
- [ ] No NaN
- [ ] Surprise values 0.1-0.8
- [ ] Gate values 0.1-0.5