"""
Phase 1: Within-Session Adaptation Training.

Trains all fast neuron θ parameters. Slow neuron inactive (context = zeros).

Episode structure (spec §Phase 1):
  - 2048 tokens from one topic, split into 8 × 256-token chunks
  - Chunks 1-6: adapt (memory + projection writes enabled)
  - Chunks 7-8: eval (read only, no writes)
  - Loss: cross-entropy on eval chunks only
  - BPTT through state tensors (mem_A, W_mod) links eval loss to adapt computations

Optimizer: AdamW, lr=3e-4, weight_decay=0.01, gradient clipping max_norm=1.0
Success metric: adaptation_benefit = loss_chunk1 - loss_chunk8 > 0 and increasing

Run with:
    python -m training.phase1_adaptation [--args]
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from model.nat_model import NATv2Model
from training.data import EpisodeDataset


@dataclass
class Phase1Config:
    """All hyperparameters for Phase 1 training."""

    # Model
    model_name: str = "Qwen/Qwen3-4B"
    layer_A: int = 9
    layer_B: int = 18

    # Episode structure
    num_episodes: int = 50_000
    batch_size: int = 4
    seq_len: int = 2048
    chunk_size: int = 256
    num_adapt_chunks: int = 6

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_episodes: int = 500

    # Memory optimization
    gradient_checkpointing: bool = False

    # Logging & checkpointing
    log_interval: int = 50
    frozen_eval_interval: int = 500
    save_interval: int = 2500

    # Paths
    output_dir: str = "checkpoints/phase1"
    data_cache_dir: str = "data/cache"
    resume_from: Optional[str] = None

    # Data (pre-tokenized .pt files; if None, loads from HuggingFace)
    data_token_files: Optional[list] = None

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "nat-v2"
    wandb_run_name: Optional[str] = None

    # Verification evaluation
    verify_interval: int = 50
    verify_episodes: int = 32

    @property
    def num_chunks(self):
        return self.seq_len // self.chunk_size


def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token prediction cross-entropy loss."""
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size),
        input_ids[:, 1:].reshape(-1),
    )


def frozen_baseline_eval(model, input_ids, chunk_size):
    """
    Compute per-chunk loss using frozen baseline (no hooks).

    Returns list of per-chunk losses (Python floats).
    """
    num_chunks = input_ids.shape[1] // chunk_size
    losses = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_ids = input_ids[:, start:end]
        outputs = model.frozen_baseline_forward(chunk_ids)
        loss = compute_loss(outputs.logits, chunk_ids)
        losses.append(loss.item())
    return losses


def create_verification_set(dataset, config, output_dir, device):
    """
    Create or load a fixed verification set of episodes.

    Returns (verify_ids, seq_len) where verify_ids is (N, seq_len) LongTensor.
    Saved to output_dir/verify_episodes.pt for reuse across resumes.
    """
    verify_path = Path(output_dir) / "verify_episodes.pt"
    if verify_path.exists():
        verify_ids = torch.load(verify_path, weights_only=True)
        print(f"Loaded verification set: {verify_ids.shape[0]} episodes from {verify_path}")
        return verify_ids

    episodes = []
    for _ in range(config.verify_episodes):
        ids, _ = dataset.sample_batch(1)
        episodes.append(ids.squeeze(0))
    verify_ids = torch.stack(episodes)  # (N, seq_len)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(verify_ids, verify_path)
    print(f"Created verification set: {verify_ids.shape[0]} episodes, saved to {verify_path}")
    return verify_ids


def run_verification_eval(model, verify_ids, config, device):
    """
    Compare NAT v2 vs frozen baseline on fixed verification episodes.

    For each episode:
      - NAT: reset state, run full episode (adapt + eval), record eval loss
      - Baseline: remove hooks, run only eval chunks, record loss, re-register hooks

    Returns dict of verify/* metrics.
    """
    num_episodes = verify_ids.shape[0]
    chunk_size = config.chunk_size
    num_chunks = config.num_chunks
    num_adapt = config.num_adapt_chunks
    batch_size = config.batch_size

    nat_eval_losses = []
    baseline_eval_losses = []

    for start_idx in range(0, num_episodes, batch_size):
        end_idx = min(start_idx + batch_size, num_episodes)
        batch_ids = verify_ids[start_idx:end_idx].to(device)
        bs = batch_ids.shape[0]

        # ---- NAT forward (adapt then eval) ----
        model.start_episode(bs, device)
        for chunk_idx in range(num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = c_start + chunk_size
            chunk_ids = batch_ids[:, c_start:c_end]

            adapt = chunk_idx < num_adapt
            with torch.no_grad():
                outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)

            if not adapt:
                loss = compute_loss(outputs.logits, chunk_ids)
                nat_eval_losses.append(loss.item())
            del outputs

        # ---- Frozen baseline (eval chunks only, hooks fully removed) ----
        model.remove_hooks()
        for chunk_idx in range(num_adapt, num_chunks):
            c_start = chunk_idx * chunk_size
            c_end = c_start + chunk_size
            chunk_ids = batch_ids[:, c_start:c_end]

            with torch.no_grad():
                outputs = model.base_model(
                    input_ids=chunk_ids, use_cache=False,
                )
            loss = compute_loss(outputs.logits, chunk_ids)
            baseline_eval_losses.append(loss.item())
            del outputs
        model.register_hooks()

    nat_loss = sum(nat_eval_losses) / len(nat_eval_losses)
    baseline_loss = sum(baseline_eval_losses) / len(baseline_eval_losses)
    improvement = baseline_loss - nat_loss
    improvement_pct = (improvement / baseline_loss * 100) if baseline_loss > 0 else 0.0

    return {
        "verify/nat_eval_loss": nat_loss,
        "verify/baseline_eval_loss": baseline_loss,
        "verify/improvement": improvement,
        "verify/improvement_pct": improvement_pct,
    }


def train_phase1(
    config: Phase1Config,
    model: Optional[NATv2Model] = None,
    dataset: Optional[EpisodeDataset] = None,
):
    """
    Run Phase 1 training loop.

    Args:
        config: Training configuration.
        model: Optional pre-created model (for testing with tiny models).
        dataset: Optional pre-created dataset (for testing).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 1 Training — device={device}, dtype={dtype}")
    print(f"Config: {json.dumps(asdict(config), indent=2, default=str)}")

    # ---- Model ----
    if model is None:
        model = NATv2Model(
            model_name=config.model_name,
            layer_A=config.layer_A,
            layer_B=config.layer_B,
            dtype=dtype,
        )
        model.to(device)

    if config.gradient_checkpointing and hasattr(
        model.base_model, "gradient_checkpointing_enable"
    ):
        model.base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    vocab_size = model.base_model.config.vocab_size
    theta_count = model.count_theta_params()
    print(f"θ parameters: {theta_count:,}")

    # ---- Optimizer ----
    theta_params = list(model.theta_params())
    optimizer = torch.optim.AdamW(
        theta_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # ---- Data ----
    if dataset is None:
        if config.data_token_files:
            dataset = EpisodeDataset.from_token_files(
                config.data_token_files, seq_len=config.seq_len,
            )
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            dataset = EpisodeDataset.from_huggingface(
                tokenizer=tokenizer,
                seq_len=config.seq_len,
                cache_dir=config.data_cache_dir,
            )

    # ---- Wandb ----
    _wandb_enabled = False
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )
            _wandb_enabled = True
            print("Wandb logging enabled")
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")

    # ---- Verification set ----
    verify_ids = None
    if config.verify_episodes > 0:
        verify_ids = create_verification_set(
            dataset, config, config.output_dir, device,
        )

    # ---- Resume from checkpoint ----
    start_episode = 1
    running_eval_loss = 0.0
    running_benefit = 0.0

    if config.resume_from:
        ckpt = torch.load(
            config.resume_from, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_episode = ckpt["episode"] + 1
        running_eval_loss = ckpt.get("running_eval_loss", 0.0)
        running_benefit = ckpt.get("running_benefit", 0.0)
        print(f"Resumed from episode {start_episode - 1}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    log_file = open(output_dir / "train_log.jsonl", "a")

    best_running_benefit = -float("inf")
    start_time = time.time()
    num_eval_chunks = config.num_chunks - config.num_adapt_chunks

    print(f"\nStarting training from episode {start_episode}...\n")

    # ================================================================
    # Training loop
    # ================================================================
    for episode in range(start_episode, config.num_episodes + 1):

        # ---- Learning rate warmup ----
        if episode <= config.warmup_episodes:
            lr_scale = episode / config.warmup_episodes
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * lr_scale

        # ---- Sample episode data ----
        input_ids, topic_indices = dataset.sample_batch(config.batch_size)
        input_ids = input_ids.to(device)

        # ---- Reset episode state ----
        model.start_episode(config.batch_size, device)
        optimizer.zero_grad()

        # ---- Process chunks ----
        chunk_losses = []
        eval_losses = []
        mem_snapshots = []  # for per-chunk gradient logging

        # Neuron diagnostic accumulators
        surprise_sum_A = 0.0
        surprise_sum_B = 0.0
        gate_sum_A = 0.0
        gate_sum_B = 0.0
        threshold_sum_A = 0.0
        threshold_sum_B = 0.0
        proj_write_sum_A = 0.0
        proj_write_sum_B = 0.0
        threshold_count = 0
        chunk_count = 0

        for chunk_idx in range(config.num_chunks):
            start = chunk_idx * config.chunk_size
            end = start + config.chunk_size
            chunk_ids = input_ids[:, start:end]

            adapt = chunk_idx < config.num_adapt_chunks
            outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)

            loss = compute_loss(outputs.logits, chunk_ids)
            chunk_losses.append(loss.item())

            if not adapt:
                eval_losses.append(loss)

            # Collect neuron diagnostics
            chunk_count += 1
            for neuron, sfx in [
                (model.fast_neuron_A, "A"),
                (model.fast_neuron_B, "B"),
            ]:
                if neuron.last_surprise is not None:
                    val = neuron.last_surprise.mean().item()
                    if sfx == "A":
                        surprise_sum_A += val
                    else:
                        surprise_sum_B += val
                if neuron.last_gate is not None:
                    val = neuron.last_gate.mean().item()
                    if sfx == "A":
                        gate_sum_A += val
                    else:
                        gate_sum_B += val
                if adapt and neuron.last_threshold is not None:
                    val = neuron.last_threshold.mean().item()
                    if sfx == "A":
                        threshold_sum_A += val
                    else:
                        threshold_sum_B += val
                if adapt and neuron.last_proj_write_mask is not None:
                    val = neuron.last_proj_write_mask.mean().item()
                    if sfx == "A":
                        proj_write_sum_A += val
                    else:
                        proj_write_sum_B += val

            if adapt and model.fast_neuron_A.last_threshold is not None:
                threshold_count += 1

            # Snapshot mem_A for per-chunk gradient logging
            if adapt and chunk_idx > 0:
                model.fast_neuron_A.mem_A.retain_grad()
                model.fast_neuron_B.mem_A.retain_grad()
                mem_snapshots.append(
                    (chunk_idx, model.fast_neuron_A.mem_A, model.fast_neuron_B.mem_A)
                )

            # Free adapt chunk graph (state tensors retain theirs)
            del outputs
            if adapt:
                del loss

        # ---- Backward + step ----
        eval_loss = torch.stack(eval_losses).mean()
        eval_loss.backward()

        # Per-chunk gradient norms
        chunk_grad_norms = {}
        chunk_grad_norms["train/grad_norm_chunk_0"] = 0.0
        for ci, mem_A, mem_B in mem_snapshots:
            gn_A = torch.norm(mem_A.grad).item() if mem_A.grad is not None else 0.0
            gn_B = torch.norm(mem_B.grad).item() if mem_B.grad is not None else 0.0
            chunk_grad_norms[f"train/grad_norm_chunk_{ci}"] = gn_A + gn_B
        del mem_snapshots

        grad_norm = torch.nn.utils.clip_grad_norm_(
            theta_params, max_norm=config.max_grad_norm
        )

        optimizer.step()

        # ---- Metrics ----
        eval_loss_val = eval_loss.item()
        adaptation_benefit = chunk_losses[0] - chunk_losses[-1]

        alpha = 0.01
        if episode == start_episode:
            running_eval_loss = eval_loss_val
            running_benefit = adaptation_benefit
        else:
            running_eval_loss = (
                (1 - alpha) * running_eval_loss + alpha * eval_loss_val
            )
            running_benefit = (
                (1 - alpha) * running_benefit + alpha * adaptation_benefit
            )

        del eval_loss, eval_losses

        # ---- Wandb logging (every episode) ----
        if _wandb_enabled:
            adapt_loss = (
                sum(chunk_losses[:config.num_adapt_chunks])
                / config.num_adapt_chunks
            )
            grad_norm_val = (
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            )
            t_count = max(threshold_count, 1)  # avoid div by zero
            wandb_metrics = {
                "train/eval_loss": eval_loss_val,
                "train/adapt_loss": adapt_loss,
                "train/adaptation_benefit": adaptation_benefit,
                "train/surprise_mean_A": surprise_sum_A / chunk_count,
                "train/surprise_mean_B": surprise_sum_B / chunk_count,
                "train/gate_mean_A": gate_sum_A / chunk_count,
                "train/gate_mean_B": gate_sum_B / chunk_count,
                "train/mem_norm_A": torch.norm(
                    model.fast_neuron_A.mem_A
                ).item(),
                "train/mem_norm_B": torch.norm(
                    model.fast_neuron_B.mem_A
                ).item(),
                "train/W_down_mod_norm_A": torch.norm(
                    model.fast_neuron_A.W_down_mod
                ).item(),
                "train/W_up_mod_norm_A": torch.norm(
                    model.fast_neuron_A.W_up_mod
                ).item(),
                "train/W_down_mod_norm_B": torch.norm(
                    model.fast_neuron_B.W_down_mod
                ).item(),
                "train/W_up_mod_norm_B": torch.norm(
                    model.fast_neuron_B.W_up_mod
                ).item(),
                "train/threshold_mean_A": threshold_sum_A / t_count,
                "train/threshold_mean_B": threshold_sum_B / t_count,
                "train/proj_write_count_A": proj_write_sum_A,
                "train/proj_write_count_B": proj_write_sum_B,
                "train/grad_norm": grad_norm_val,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/episode": episode,
            }
            wandb_metrics.update(chunk_grad_norms)
            wandb.log(wandb_metrics, step=episode)

        # ---- Logging ----
        if episode % config.log_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode - start_episode + 1) / elapsed

            metrics = {
                "episode": episode,
                "eval_loss": eval_loss_val,
                "running_eval_loss": running_eval_loss,
                "adaptation_benefit": adaptation_benefit,
                "running_benefit": running_benefit,
                "chunk_losses": chunk_losses,
                "grad_norm": (
                    grad_norm.item()
                    if torch.is_tensor(grad_norm)
                    else grad_norm
                ),
                "mem_A_norm_A": torch.norm(
                    model.fast_neuron_A.mem_A
                ).item(),
                "mem_A_norm_B": torch.norm(
                    model.fast_neuron_B.mem_A
                ).item(),
                "lr": optimizer.param_groups[0]["lr"],
                "eps_per_sec": eps_per_sec,
                "elapsed_min": elapsed / 60,
            }

            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()

            print(
                f"[{episode:>6}] "
                f"eval={eval_loss_val:.4f} "
                f"benefit={adaptation_benefit:+.4f} "
                f"(avg={running_benefit:+.4f}) "
                f"grad={metrics['grad_norm']:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.1e} "
                f"({eps_per_sec:.2f} ep/s)"
            )

        # ---- Periodic frozen baseline comparison ----
        if episode % config.frozen_eval_interval == 0:
            with torch.no_grad():
                frozen_losses = frozen_baseline_eval(
                    model, input_ids, config.chunk_size,
                )
            frozen_eval = (
                sum(frozen_losses[config.num_adapt_chunks:]) / num_eval_chunks
            )
            adapted_improvement = frozen_eval - eval_loss_val

            print(
                f"  Frozen baseline: eval={frozen_eval:.4f}, "
                f"adapted_delta={adapted_improvement:+.4f}, "
                f"frozen_per_chunk="
                f"{[f'{l:.3f}' for l in frozen_losses]}"
            )

            # Log frozen comparison
            frozen_metrics = {
                "episode": episode,
                "type": "frozen_eval",
                "frozen_eval_loss": frozen_eval,
                "adapted_eval_loss": eval_loss_val,
                "adapted_improvement": adapted_improvement,
                "frozen_chunk_losses": frozen_losses,
            }
            log_file.write(json.dumps(frozen_metrics) + "\n")
            log_file.flush()

        # ---- Verification evaluation ----
        if (
            verify_ids is not None
            and config.verify_interval > 0
            and episode % config.verify_interval == 0
        ):
            verify_metrics = run_verification_eval(
                model, verify_ids, config, device,
            )
            print(
                f"  Verify: NAT={verify_metrics['verify/nat_eval_loss']:.4f}, "
                f"baseline={verify_metrics['verify/baseline_eval_loss']:.4f}, "
                f"improvement={verify_metrics['verify/improvement']:+.4f} "
                f"({verify_metrics['verify/improvement_pct']:+.1f}%)"
            )
            if _wandb_enabled:
                wandb.log(verify_metrics, step=episode)

            # Also log to local file
            verify_log = {"episode": episode, "type": "verification"}
            verify_log.update(verify_metrics)
            log_file.write(json.dumps(verify_log) + "\n")
            log_file.flush()

        # ---- Checkpointing ----
        if episode % config.save_interval == 0:
            save_path = output_dir / f"checkpoint_{episode}.pt"
            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": {
                        k: v
                        for k, v in model.state_dict().items()
                        if "base_model" not in k
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "running_eval_loss": running_eval_loss,
                    "running_benefit": running_benefit,
                },
                save_path,
            )
            print(f"  Saved checkpoint: {save_path}")

            if running_benefit > best_running_benefit:
                best_running_benefit = running_benefit
                best_path = output_dir / "best.pt"
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": {
                            k: v
                            for k, v in model.state_dict().items()
                            if "base_model" not in k
                        },
                        "running_benefit": running_benefit,
                    },
                    best_path,
                )
                print(f"  New best! benefit={running_benefit:+.4f}")

    log_file.close()
    if _wandb_enabled:
        wandb.finish()
    print(f"\nTraining complete. {config.num_episodes} episodes.")
    print(f"Best running adaptation benefit: {best_running_benefit:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="NAT v2 Phase 1 Training")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--num-episodes", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-episodes", type=int, default=500)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints/phase1")
    parser.add_argument("--data-cache-dir", default="data/cache")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--frozen-eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=2500)
    parser.add_argument("--data-token-files", nargs="+", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="nat-v2")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--verify-interval", type=int, default=50)
    parser.add_argument("--verify-episodes", type=int, default=32)
    args = parser.parse_args()

    config = Phase1Config(
        model_name=args.model_name,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_episodes=args.warmup_episodes,
        gradient_checkpointing=args.gradient_checkpointing,
        output_dir=args.output_dir,
        data_cache_dir=args.data_cache_dir,
        resume_from=args.resume_from,
        log_interval=args.log_interval,
        frozen_eval_interval=args.frozen_eval_interval,
        save_interval=args.save_interval,
        data_token_files=args.data_token_files,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        verify_interval=args.verify_interval,
        verify_episodes=args.verify_episodes,
    )

    train_phase1(config)


if __name__ == "__main__":
    main()
