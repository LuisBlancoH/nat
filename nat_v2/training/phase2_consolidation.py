"""
Phase 2: Across-Session Consolidation Training.

Activates the slow neuron to learn consolidation: preserving useful
adaptations across domain shifts.

Episode structure (spec §Phase 2):
  - 13 windows × 2048 tokens × 8 chunks each = 104 forward passes
  - Windows 1-5: domain A, 6-10: domain B, 11-13: domain A return
  - Fast neuron mem_A resets each window, W_mod persists
  - Slow neuron fires every 16 chunks (6 firings per episode)
  - Per-window BPTT with detach at window boundaries

Loss: cross-entropy on eval chunks (6-7) of each window.
Training signal: windows 11-13 eval loss trains consolidation network
to produce useful writes when returning to a previously-seen domain.

Optimizer: Two groups — slow neuron (3e-4), fast neurons (1e-5 fine-tuning).

Run with:
    python -m training.phase2_consolidation [--args]
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from model.nat_model import NATv2Model
from training.data import EpisodeDataset
from training.phase1_adaptation import compute_loss


@dataclass
class Phase2Config:
    """All hyperparameters for Phase 2 training."""

    # Model
    model_name: str = "Qwen/Qwen3-4B"
    layer_A: int = 9
    layer_B: int = 18

    # Episode structure
    num_episodes: int = 5_000
    batch_size: int = 4
    window_len: int = 2048
    chunk_size: int = 256
    num_adapt_chunks: int = 6
    windows_A: int = 8       # 5 early + 3 late
    windows_B: int = 5
    slow_fire_interval: int = 16

    # Optimizer
    slow_lr: float = 3e-4
    fast_lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_episodes: int = 100

    # Logging & checkpointing
    log_interval: int = 10
    save_interval: int = 500
    verify_interval: int = 50

    # Paths
    output_dir: str = "checkpoints/phase2"
    phase1_checkpoint: str = "checkpoints/phase1/best.pt"
    data_cache_dir: str = "data/cache"
    resume_from: Optional[str] = None

    # Data
    data_token_files: Optional[list] = None

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "nat-v2"
    wandb_run_name: Optional[str] = None

    # Verification
    verify_episodes: int = 16

    @property
    def num_chunks_per_window(self):
        return self.window_len // self.chunk_size

    @property
    def num_windows(self):
        return 5 + self.windows_B + 3  # A_early + B + A_late


def load_phase1_checkpoint(model: NATv2Model, path: str, device: torch.device):
    """
    Load Phase 1 best checkpoint for fast neuron weights.
    Slow neuron starts fresh (random init from model construction).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model_state = state["model_state_dict"]

    # Load only fast neuron parameters (slow neuron stays at random init)
    fast_keys = {
        k: v for k, v in model_state.items()
        if k.startswith("fast_neuron_A.") or k.startswith("fast_neuron_B.")
    }

    missing, unexpected = model.load_state_dict(fast_keys, strict=False)
    # Expected: all slow neuron + base model keys are "missing" (not loaded)
    loaded_count = len(fast_keys)
    print(f"Loaded Phase 1 checkpoint: {loaded_count} fast neuron params from {path}")
    if state.get("episode"):
        print(f"  Phase 1 episode: {state['episode']}")
    if state.get("running_benefit"):
        print(f"  Phase 1 running benefit: {state['running_benefit']:+.4f}")


def run_episode(
    model: NATv2Model,
    batch: dict,
    config: Phase2Config,
    device: torch.device,
):
    """
    Run one Phase 2 training episode (13 windows).

    Returns dict of metrics (all Python floats, detached).
    Gradients are accumulated across all windows (caller does optimizer step).
    """
    windows = batch['windows']
    domain_labels = batch['domain_labels']
    num_chunks = config.num_chunks_per_window
    chunk_size = config.chunk_size
    num_adapt = config.num_adapt_chunks
    batch_size = windows[0].shape[0]

    # Track per-domain losses using domain_labels
    # Find domain phase boundaries: first A block = early, B block, last A block = late
    losses_A_early = []
    losses_B = []
    losses_A_late = []
    per_window_losses = []

    # Determine phase for each window: A_early until first B, then B, then A_late
    seen_B = False
    window_phases = []
    for label in domain_labels:
        if label == 'B':
            seen_B = True
            window_phases.append('B')
        elif seen_B:
            window_phases.append('A_late')
        else:
            window_phases.append('A_early')

    for w_idx, (window_ids, label) in enumerate(zip(windows, domain_labels)):
        window_ids = window_ids.to(device)

        # Reset fast neuron memory for new window (W_mod, context persist)
        model.start_window(batch_size, device)

        window_eval_losses = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            chunk_ids = window_ids[:, start:end]

            adapt = chunk_idx < num_adapt
            outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)

            loss = compute_loss(outputs.logits, chunk_ids)

            if not adapt:
                window_eval_losses.append(loss)

            del outputs
            if adapt:
                del loss

        # Compute window eval loss and backward (accumulates grads)
        if window_eval_losses:
            window_eval_loss = torch.stack(window_eval_losses).mean()
            window_eval_loss.backward()
            loss_val = window_eval_loss.item()
        else:
            loss_val = 0.0

        per_window_losses.append(loss_val)

        # Categorize by domain phase
        phase = window_phases[w_idx]
        if phase == 'A_early':
            losses_A_early.append(loss_val)
        elif phase == 'B':
            losses_B.append(loss_val)
        else:
            losses_A_late.append(loss_val)

        # Detach all persistent state at window boundary
        model.detach_all_state()

        del window_eval_losses
        if 'window_eval_loss' in dir():
            del window_eval_loss

    # Compute metrics
    loss_A_early = sum(losses_A_early) / max(len(losses_A_early), 1)
    loss_B = sum(losses_B) / max(len(losses_B), 1)
    loss_A_late = sum(losses_A_late) / max(len(losses_A_late), 1)
    forgetting_ratio = loss_A_late / (loss_A_early + 1e-8)
    consolidation_benefit = loss_A_early - loss_A_late

    metrics = {
        "loss_A_early": loss_A_early,
        "loss_B": loss_B,
        "loss_A_late": loss_A_late,
        "forgetting_ratio": forgetting_ratio,
        "consolidation_benefit": consolidation_benefit,
        "per_window_losses": per_window_losses,
    }

    # Slow neuron state health
    if model.slow_neuron.mem_A is not None:
        metrics["slow_context_norm"] = torch.norm(
            model.slow_neuron.default_context
        ).item()
        metrics["slow_mem_A_norm"] = torch.norm(
            model.slow_neuron.mem_A
        ).item()

    # Fast neuron W_mod norms (end of episode)
    if model.enable_neuron_A and model.fast_neuron_A.W_down_mod is not None:
        metrics["W_mod_norm_A"] = (
            torch.norm(model.fast_neuron_A.W_down_mod).item()
            + torch.norm(model.fast_neuron_A.W_up_mod).item()
        )
    metrics["W_mod_norm_B"] = (
        torch.norm(model.fast_neuron_B.W_down_mod).item()
        + torch.norm(model.fast_neuron_B.W_up_mod).item()
    )

    return metrics


def run_verification(
    model: NATv2Model,
    verify_dataset: EpisodeDataset,
    config: Phase2Config,
    device: torch.device,
):
    """
    Run verification on held-out topics with the same 13-window structure.
    Returns dict of verification metrics.
    """
    num_verify = config.verify_episodes
    all_forgetting = []
    all_benefit = []
    all_loss_A_early = []
    all_loss_A_late = []
    all_loss_B = []

    for _ in range(num_verify):
        try:
            batch = verify_dataset.sample_phase2_batch(
                batch_size=1,
                windows_A=config.windows_A,
                windows_B=config.windows_B,
                window_len=config.window_len,
            )
        except ValueError:
            break

        model.start_episode(1, device)
        model.slow_neuron_active = True

        with torch.no_grad():
            metrics = run_episode(model, batch, config, device)

        all_forgetting.append(metrics["forgetting_ratio"])
        all_benefit.append(metrics["consolidation_benefit"])
        all_loss_A_early.append(metrics["loss_A_early"])
        all_loss_A_late.append(metrics["loss_A_late"])
        all_loss_B.append(metrics["loss_B"])

    if not all_forgetting:
        return {}

    return {
        "verify/forgetting_ratio": sum(all_forgetting) / len(all_forgetting),
        "verify/consolidation_benefit": sum(all_benefit) / len(all_benefit),
        "verify/loss_A_early": sum(all_loss_A_early) / len(all_loss_A_early),
        "verify/loss_A_late": sum(all_loss_A_late) / len(all_loss_A_late),
        "verify/loss_B": sum(all_loss_B) / len(all_loss_B),
        "verify/num_episodes": len(all_forgetting),
    }


def train_phase2(
    config: Phase2Config,
    model: Optional[NATv2Model] = None,
    dataset: Optional[EpisodeDataset] = None,
):
    """
    Run Phase 2 training loop.

    Args:
        config: Training configuration.
        model: Optional pre-created model (for testing with tiny models).
        dataset: Optional pre-created dataset (for testing).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 2 Training — device={device}, dtype={dtype}")
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

        # Load Phase 1 checkpoint
        load_phase1_checkpoint(model, config.phase1_checkpoint, device)

    # Activate slow neuron
    model.slow_neuron_active = True

    theta_count = model.count_theta_params()
    print(f"θ parameters: {theta_count:,}")

    # ---- Optimizer: two parameter groups ----
    slow_params = list(model.slow_neuron.parameters())
    fast_params = (
        list(model.fast_neuron_A.parameters())
        + list(model.fast_neuron_B.parameters())
    )

    optimizer = torch.optim.AdamW([
        {'params': slow_params, 'lr': config.slow_lr},
        {'params': fast_params, 'lr': config.fast_lr},
    ], weight_decay=config.weight_decay)

    # Cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_episodes, eta_min=1e-6,
    )

    # ---- Data ----
    if dataset is None:
        if config.data_token_files:
            dataset = EpisodeDataset.from_token_files(
                config.data_token_files, seq_len=config.window_len,
            )
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            dataset = EpisodeDataset.from_huggingface(
                tokenizer=tokenizer,
                seq_len=config.window_len,
                cache_dir=config.data_cache_dir,
            )

    # ---- Wandb ----
    _wandb_enabled = False
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or "phase2",
                config=asdict(config),
            )
            _wandb_enabled = True
            print("Wandb logging enabled")
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")

    # ---- Verification set (held-out topics) ----
    verify_dataset = None
    if config.verify_episodes > 0:
        verify_dataset = dataset.split_topics()

    # ---- Resume from checkpoint ----
    start_episode = 1
    if config.resume_from:
        ckpt = torch.load(
            config.resume_from, map_location=device, weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_episode = ckpt["episode"] + 1
        print(f"Resumed from episode {start_episode - 1}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    log_file = open(output_dir / "train_log.jsonl", "a")

    best_forgetting_ratio = float("inf")
    start_time = time.time()

    print(f"\nStarting Phase 2 training from episode {start_episode}...\n")

    # ================================================================
    # Training loop
    # ================================================================
    for episode in range(start_episode, config.num_episodes + 1):

        # ---- Learning rate warmup ----
        if episode <= config.warmup_episodes:
            lr_scale = episode / config.warmup_episodes
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * lr_scale
            # Store initial lr for warmup reference
            if episode == 1:
                for pg in optimizer.param_groups:
                    pg["initial_lr"] = pg["lr"]

        # ---- Sample episode data ----
        try:
            batch = dataset.sample_phase2_batch(
                batch_size=config.batch_size,
                windows_A=config.windows_A,
                windows_B=config.windows_B,
                window_len=config.window_len,
            )
        except ValueError as e:
            print(f"Warning: failed to sample batch: {e}")
            continue

        # ---- Reset episode state ----
        model.start_episode(config.batch_size, device)
        model.slow_neuron_active = True
        optimizer.zero_grad()

        # ---- Run episode (accumulates gradients across 13 windows) ----
        metrics = run_episode(model, batch, config, device)

        # ---- Clip + step ----
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.theta_params()), max_norm=config.max_grad_norm,
        )
        optimizer.step()
        if episode > config.warmup_episodes:
            scheduler.step()

        grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

        # ---- Wandb logging ----
        if _wandb_enabled:
            wandb_metrics = {
                "train/loss_A_early": metrics["loss_A_early"],
                "train/loss_A_late": metrics["loss_A_late"],
                "train/loss_B": metrics["loss_B"],
                "train/forgetting_ratio": metrics["forgetting_ratio"],
                "train/consolidation_benefit": metrics["consolidation_benefit"],
                "train/grad_norm": grad_norm_val,
                "train/slow_lr": optimizer.param_groups[0]["lr"],
                "train/fast_lr": optimizer.param_groups[1]["lr"],
                "train/episode": episode,
                "fast/W_mod_norm_A": metrics.get("W_mod_norm_A", 0.0),
                "fast/W_mod_norm_B": metrics["W_mod_norm_B"],
            }
            if "slow_context_norm" in metrics:
                wandb_metrics["slow/context_norm"] = metrics["slow_context_norm"]
                wandb_metrics["slow/mem_A_norm"] = metrics["slow_mem_A_norm"]

            # Per-window losses
            for i, wl in enumerate(metrics["per_window_losses"]):
                wandb_metrics[f"train/window_{i}_loss"] = wl

            wandb.log(wandb_metrics, step=episode)

        # ---- Console + file logging ----
        if episode % config.log_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode - start_episode + 1) / elapsed

            log_entry = {
                "episode": episode,
                "loss_A_early": metrics["loss_A_early"],
                "loss_A_late": metrics["loss_A_late"],
                "loss_B": metrics["loss_B"],
                "forgetting_ratio": metrics["forgetting_ratio"],
                "consolidation_benefit": metrics["consolidation_benefit"],
                "grad_norm": grad_norm_val,
                "W_mod_norm_A": metrics.get("W_mod_norm_A", 0.0),
                "W_mod_norm_B": metrics["W_mod_norm_B"],
                "per_window_losses": metrics["per_window_losses"],
                "eps_per_sec": eps_per_sec,
                "elapsed_min": elapsed / 60,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            print(
                f"[{episode:>6}] "
                f"A_early={metrics['loss_A_early']:.4f} "
                f"B={metrics['loss_B']:.4f} "
                f"A_late={metrics['loss_A_late']:.4f} "
                f"forget={metrics['forgetting_ratio']:.3f} "
                f"benefit={metrics['consolidation_benefit']:+.4f} "
                f"grad={grad_norm_val:.2f} "
                f"({eps_per_sec:.2f} ep/s)"
            )

        # ---- Verification ----
        if (
            verify_dataset is not None
            and config.verify_interval > 0
            and episode % config.verify_interval == 0
        ):
            verify_metrics = run_verification(
                model, verify_dataset, config, device,
            )
            if verify_metrics:
                print(
                    f"  Verify: forget={verify_metrics['verify/forgetting_ratio']:.3f}, "
                    f"benefit={verify_metrics['verify/consolidation_benefit']:+.4f}, "
                    f"A_early={verify_metrics['verify/loss_A_early']:.4f}, "
                    f"A_late={verify_metrics['verify/loss_A_late']:.4f}"
                )
                if _wandb_enabled:
                    wandb.log(verify_metrics, step=episode)

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
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                save_path,
            )
            print(f"  Saved checkpoint: {save_path}")

            if metrics["forgetting_ratio"] < best_forgetting_ratio:
                best_forgetting_ratio = metrics["forgetting_ratio"]
                best_path = output_dir / "best.pt"
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": {
                            k: v
                            for k, v in model.state_dict().items()
                            if "base_model" not in k
                        },
                        "forgetting_ratio": best_forgetting_ratio,
                    },
                    best_path,
                )
                print(f"  New best! forgetting_ratio={best_forgetting_ratio:.3f}")

    log_file.close()
    if _wandb_enabled:
        wandb.finish()
    print(f"\nPhase 2 training complete. {config.num_episodes} episodes.")
    print(f"Best forgetting ratio: {best_forgetting_ratio:.3f}")


def main():
    parser = argparse.ArgumentParser(description="NAT v2 Phase 2 Training")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--num-episodes", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--slow-lr", type=float, default=3e-4)
    parser.add_argument("--fast-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-episodes", type=int, default=100)
    parser.add_argument("--output-dir", default="checkpoints/phase2")
    parser.add_argument("--phase1-checkpoint", default="checkpoints/phase1/best.pt")
    parser.add_argument("--data-cache-dir", default="data/cache")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--verify-interval", type=int, default=50)
    parser.add_argument("--data-token-files", nargs="+", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="nat-v2")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--verify-episodes", type=int, default=16)
    args = parser.parse_args()

    config = Phase2Config(
        model_name=args.model_name,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        slow_lr=args.slow_lr,
        fast_lr=args.fast_lr,
        warmup_episodes=args.warmup_episodes,
        output_dir=args.output_dir,
        phase1_checkpoint=args.phase1_checkpoint,
        data_cache_dir=args.data_cache_dir,
        resume_from=args.resume_from,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        verify_interval=args.verify_interval,
        data_token_files=args.data_token_files,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        verify_episodes=args.verify_episodes,
    )

    train_phase2(config)


if __name__ == "__main__":
    main()
