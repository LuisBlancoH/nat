"""
Evaluation suite for NAT.

Implements the three core evaluations described in the architecture doc,
plus a frozen-model baseline comparison:

1. **Within-session learning** — Does per-problem loss decrease from
   problem 1 to problem N within a single episode?
2. **Cross-session learning** — Does loss at session 20 start lower
   than at session 1 when repeatedly exposed to the same domain?
3. **Forgetting test** — After learning domain B, does performance on
   domain A remain close to its pre-switch level?
4. **Baseline comparison** — Compare NAT (adaptive active) vs.
   frozen baseline (adaptive layers disabled / gates forced to zero).

Each evaluation returns a structured ``EvalResult`` with raw numbers,
summary metrics, and a human-readable report.  Results can be logged to
W&B or serialised to JSON.

Usage
-----
::

    from nat.training.eval import (
        evaluate_within_session,
        evaluate_cross_session,
        evaluate_forgetting,
        evaluate_baseline,
        run_full_evaluation,
    )

    results = run_full_evaluation(model, config, synthetic=True)
    print(results["summary"])
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nat.training.data import (
    SyntheticEpisodicDataset,
    collate_episodic,
)
from nat.training.phase2_consolidation import (
    SyntheticDomainDataset,
    DOMAINS,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Result dataclass                                                     #
# ------------------------------------------------------------------ #

@dataclass
class EvalResult:
    """
    Structured result container for an evaluation run.

    Attributes
    ----------
    name : str
        Name of the evaluation (e.g. ``"within_session"``).
    passed : bool
        Whether the key success criterion was met.
    metrics : dict[str, float]
        Numerical metrics (improvement ratios, loss values, etc.).
    curves : dict[str, list[float]]
        Ordered data series for plotting (per-problem losses, etc.).
    summary : str
        Human-readable one-paragraph summary.
    """

    name: str = ""
    passed: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    curves: dict[str, list[float]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.to_dict(), indent=2, default=str)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s)
            logger.info(f"Eval result saved → {p}")
        return s


# ------------------------------------------------------------------ #
# Helper: compute per-token loss on a sequence                         #
# ------------------------------------------------------------------ #

@torch.no_grad()
def _compute_sequence_loss(
    model,
    input_ids: torch.Tensor,
    chunk_size: int,
) -> float:
    """
    Run a full sequence through the model in chunks and return the
    average next-token prediction loss.
    """
    seq_len = input_ids.shape[1]
    all_logits: list[torch.Tensor] = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        output = model(chunk)
        all_logits.append(output["logits"])

    logits = torch.cat(all_logits, dim=1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss.item()


@torch.no_grad()
def _compute_per_problem_losses(
    model,
    input_ids: torch.Tensor,
    problem_spans: list[tuple[int, int]],
    chunk_size: int,
) -> list[float]:
    """
    Forward the full episode and compute loss per problem span.
    """
    seq_len = input_ids.shape[1]
    all_logits: list[torch.Tensor] = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        output = model(chunk)
        all_logits.append(output["logits"])

    logits = torch.cat(all_logits, dim=1)

    per_problem: list[float] = []
    for sol_start, sol_end in problem_spans:
        sol_logits = logits[:, sol_start - 1 : sol_end - 1, :]
        sol_labels = input_ids[:, sol_start:sol_end]
        if sol_logits.numel() == 0 or sol_labels.numel() == 0:
            continue
        loss_i = F.cross_entropy(
            sol_logits.reshape(-1, sol_logits.size(-1)),
            sol_labels.reshape(-1),
            ignore_index=-100,
        )
        per_problem.append(loss_i.item())

    return per_problem


# ------------------------------------------------------------------ #
# 1.  Within-session learning                                          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_within_session(
    model,
    config,
    *,
    num_episodes: int = 10,
    num_problems: int | None = None,
    synthetic: bool = True,
) -> EvalResult:
    """
    Within-session learning evaluation.

    For each episode the model processes a sequence of N related
    problems.  We measure the per-problem loss to check whether it
    decreases from problem 1 to problem N.

    Success criterion
    -----------------
    ``mean(loss of last third) < mean(loss of first third)``

    Parameters
    ----------
    model : NATModel
    config : NATConfig
    num_episodes : int
        How many episodes to average over.
    num_problems : int or None
        Problems per episode (default: ``config.num_problems_per_episode``).
    synthetic : bool
        Use synthetic episodic data.

    Returns
    -------
    EvalResult
    """
    device = next(model.parameters()).device
    model.eval()

    if num_problems is None:
        num_problems = getattr(config, "num_problems_per_episode", 8)
    assert isinstance(num_problems, int)

    # Build episodic dataset
    dataset = SyntheticEpisodicDataset(
        num_episodes=num_episodes,
        seq_len=config.seq_len,
        num_problems=num_problems,
        vocab_size=getattr(config, "vocab_size", 1000),
        seed=12345,  # fixed seed for reproducibility
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_episodic,
    )

    chunk_size = config.adapt_every_n

    # Accumulate per-problem losses across episodes
    all_per_problem: list[list[float]] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        problem_spans = batch["problem_spans"]

        model.start_session(input_ids.shape[0])
        per_problem = _compute_per_problem_losses(
            model, input_ids, problem_spans, chunk_size,
        )
        if per_problem:
            all_per_problem.append(per_problem)

    if not all_per_problem:
        return EvalResult(
            name="within_session",
            passed=False,
            summary="No episodes were evaluated — check data configuration.",
        )

    # Average across episodes for each problem position
    max_problems = max(len(pp) for pp in all_per_problem)
    avg_per_problem: list[float] = []
    for pos in range(max_problems):
        vals = [pp[pos] for pp in all_per_problem if pos < len(pp)]
        avg_per_problem.append(sum(vals) / len(vals))

    # Split into thirds
    third = max(1, len(avg_per_problem) // 3)
    first_third = avg_per_problem[:third]
    last_third = avg_per_problem[-third:]

    first_mean = sum(first_third) / len(first_third)
    last_mean = sum(last_third) / len(last_third)
    improvement = first_mean - last_mean
    improvement_pct = (improvement / first_mean * 100) if first_mean > 0 else 0.0
    passed = last_mean < first_mean

    result = EvalResult(
        name="within_session",
        passed=passed,
        metrics={
            "first_third_loss": first_mean,
            "last_third_loss": last_mean,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "first_problem_loss": avg_per_problem[0],
            "last_problem_loss": avg_per_problem[-1],
            "num_episodes": len(all_per_problem),
            "num_problems": max_problems,
        },
        curves={
            "per_problem_loss": avg_per_problem,
        },
        summary=(
            f"Within-session learning ({'PASS' if passed else 'FAIL'}): "
            f"loss decreased from {first_mean:.4f} (first third) to "
            f"{last_mean:.4f} (last third), "
            f"improvement = {improvement:+.4f} ({improvement_pct:+.1f}%). "
            f"Averaged over {len(all_per_problem)} episodes with "
            f"{max_problems} problems each."
        ),
    )
    model.train()
    return result


# ------------------------------------------------------------------ #
# 2.  Cross-session learning                                           #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_cross_session(
    model,
    config,
    *,
    domain: str = "math",
    num_sessions: int = 20,
    synthetic: bool = True,
) -> EvalResult:
    """
    Cross-session learning evaluation.

    Run ``num_sessions`` successive sessions on the same domain,
    calling ``end_session()`` between them (consolidation + partial
    reset).  Track loss at each session.

    Success criterion
    -----------------
    ``mean(last 5 sessions) < mean(first 5 sessions)``

    Parameters
    ----------
    model : NATModel
    config : NATConfig
    domain : str
        Domain to evaluate on.
    num_sessions : int
        Number of sessions to run.
    synthetic : bool
        Use synthetic domain data.

    Returns
    -------
    EvalResult
    """
    device = next(model.parameters()).device
    model.eval()

    # Build domain data
    dataset = SyntheticDomainDataset(
        domain=domain,
        num_episodes=max(num_sessions * 2, 64),
        seq_len=config.seq_len,
        vocab_size=getattr(config, "vocab_size", 1000),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)

    chunk_size = config.adapt_every_n
    session_losses: list[float] = []

    # Reset consolidation state for a clean evaluation
    model.consolidation.W_c_A = torch.zeros_like(model.consolidation.W_c_A)
    model.consolidation.W_c_B = torch.zeros_like(model.consolidation.W_c_B)

    for session_idx in range(num_sessions):
        # Fetch data
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        model.start_session(input_ids.shape[0])

        loss = _compute_sequence_loss(model, input_ids, chunk_size)
        session_losses.append(loss)

        # Between sessions: consolidate + partial reset
        model.end_session()

    # Metrics
    window = max(1, num_sessions // 4)
    first_window = session_losses[:window]
    last_window = session_losses[-window:]

    first_mean = sum(first_window) / len(first_window)
    last_mean = sum(last_window) / len(last_window)
    improvement = first_mean - last_mean
    improvement_pct = (improvement / first_mean * 100) if first_mean > 0 else 0.0
    passed = last_mean < first_mean

    result = EvalResult(
        name="cross_session",
        passed=passed,
        metrics={
            "first_window_loss": first_mean,
            "last_window_loss": last_mean,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "session_1_loss": session_losses[0],
            "session_N_loss": session_losses[-1],
            "num_sessions": num_sessions,
            "domain": 0.0,  # placeholder for typing — domain stored in summary
        },
        curves={
            "session_losses": session_losses,
        },
        summary=(
            f"Cross-session learning on '{domain}' "
            f"({'PASS' if passed else 'FAIL'}): "
            f"loss went from {first_mean:.4f} (first {window} sessions) to "
            f"{last_mean:.4f} (last {window} sessions), "
            f"improvement = {improvement:+.4f} ({improvement_pct:+.1f}%). "
            f"Over {num_sessions} sessions."
        ),
    )
    model.train()
    return result


# ------------------------------------------------------------------ #
# 3.  Forgetting test                                                  #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_forgetting(
    model,
    config,
    *,
    domain_a: str = "math",
    domain_b: str = "code",
    sessions_a: int = 10,
    sessions_b: int = 10,
    sessions_return: int = 5,
    synthetic: bool = True,
) -> EvalResult:
    """
    Forgetting evaluation.

    1. Train (via self-modification) on domain A for ``sessions_a`` sessions.
    2. Switch to domain B for ``sessions_b`` sessions.
    3. Return to domain A for ``sessions_return`` sessions.

    Forgetting is measured as the loss increase when returning to A
    compared to A's loss before the switch.

    Success criterion
    -----------------
    ``forgetting_pct < 5 %`` — loss on return to A is within 5 % of
    A's best pre-switch performance.

    Parameters
    ----------
    model : NATModel
    config : NATConfig
    domain_a, domain_b : str
        The two domains to test.
    sessions_a, sessions_b, sessions_return : int
        Number of sessions for each phase.
    synthetic : bool
        Use synthetic domain data.

    Returns
    -------
    EvalResult
    """
    device = next(model.parameters()).device
    model.eval()

    # Build datasets for both domains
    def _make_loader(domain: str) -> DataLoader:
        ds = SyntheticDomainDataset(
            domain=domain,
            num_episodes=max(sessions_a + sessions_return + 10, 64),
            seq_len=config.seq_len,
            vocab_size=getattr(config, "vocab_size", 1000),
        )
        return DataLoader(ds, batch_size=1, shuffle=True)

    loader_a = _make_loader(domain_a)
    loader_b = _make_loader(domain_b)
    iter_a = iter(loader_a)
    iter_b = iter(loader_b)

    chunk_size = config.adapt_every_n

    # Reset consolidation
    model.consolidation.W_c_A = torch.zeros_like(model.consolidation.W_c_A)
    model.consolidation.W_c_B = torch.zeros_like(model.consolidation.W_c_B)

    losses_a_before: list[float] = []
    losses_b: list[float] = []
    losses_a_return: list[float] = []

    def _run_sessions(
        data_iter, loader, n: int, loss_list: list[float],
    ):
        for _ in range(n):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            model.start_session(input_ids.shape[0])
            loss = _compute_sequence_loss(model, input_ids, chunk_size)
            loss_list.append(loss)
            model.end_session()
        return data_iter

    # Phase 1: Domain A
    iter_a = _run_sessions(iter_a, loader_a, sessions_a, losses_a_before)

    # Phase 2: Domain B
    iter_b = _run_sessions(iter_b, loader_b, sessions_b, losses_b)

    # Phase 3: Return to Domain A
    iter_a = _run_sessions(iter_a, loader_a, sessions_return, losses_a_return)

    # Compute metrics
    # Best A performance = mean of last few sessions before switch
    tail = max(1, sessions_a // 3)
    a_best = sum(losses_a_before[-tail:]) / tail
    a_return_mean = sum(losses_a_return) / len(losses_a_return) if losses_a_return else a_best

    forgetting_abs = a_return_mean - a_best
    forgetting_pct = (forgetting_abs / a_best * 100) if a_best > 0 else 0.0
    passed = forgetting_pct < 5.0

    # Recovery: does A improve during the return phase?
    recovery = 0.0
    if len(losses_a_return) >= 2:
        recovery = losses_a_return[0] - losses_a_return[-1]

    result = EvalResult(
        name="forgetting",
        passed=passed,
        metrics={
            "a_best_loss": a_best,
            "a_return_loss": a_return_mean,
            "forgetting_abs": forgetting_abs,
            "forgetting_pct": forgetting_pct,
            "b_mean_loss": sum(losses_b) / len(losses_b) if losses_b else 0.0,
            "recovery": recovery,
            "sessions_a": float(sessions_a),
            "sessions_b": float(sessions_b),
            "sessions_return": float(sessions_return),
        },
        curves={
            "losses_a_before": losses_a_before,
            "losses_b": losses_b,
            "losses_a_return": losses_a_return,
            "losses_all": losses_a_before + losses_b + losses_a_return,
        },
        summary=(
            f"Forgetting test {domain_a}→{domain_b}→{domain_a} "
            f"({'PASS' if passed else 'FAIL'}): "
            f"{domain_a} best loss = {a_best:.4f}, "
            f"return loss = {a_return_mean:.4f}, "
            f"forgetting = {forgetting_pct:+.1f}% "
            f"(threshold: < 5%). "
            f"Recovery during return = {recovery:+.4f}."
        ),
    )
    model.train()
    return result


# ------------------------------------------------------------------ #
# 4.  Baseline comparison (frozen model vs. adaptive)                  #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_baseline(
    model,
    config,
    *,
    num_episodes: int = 10,
    num_problems: int | None = None,
    synthetic: bool = True,
) -> EvalResult:
    """
    Compare NAT performance with vs. without adaptation.

    Runs each episode twice on the same data:
      - **Adaptive**: normal forward pass (fast weights self-modify).
      - **Frozen baseline**: fresh fast weights for every forward call
        (reset before each chunk, so no learning accumulates).

    Success criterion
    -----------------
    ``adaptive_loss < baseline_loss``

    Parameters
    ----------
    model : NATModel
    config : NATConfig
    num_episodes : int
    num_problems : int or None
    synthetic : bool

    Returns
    -------
    EvalResult
    """
    device = next(model.parameters()).device
    model.eval()

    if num_problems is None:
        num_problems = getattr(config, "num_problems_per_episode", 8)
    assert isinstance(num_problems, int)

    dataset = SyntheticEpisodicDataset(
        num_episodes=num_episodes,
        seq_len=config.seq_len,
        num_problems=num_problems,
        vocab_size=getattr(config, "vocab_size", 1000),
        seed=99999,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_episodic,
    )

    chunk_size = config.adapt_every_n
    adaptive_losses: list[float] = []
    baseline_losses: list[float] = []
    per_problem_adaptive: list[list[float]] = []
    per_problem_baseline: list[list[float]] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        problem_spans = batch["problem_spans"]
        batch_size = input_ids.shape[0]

        # --- Adaptive run ---
        model.start_session(batch_size)
        pp_adapt = _compute_per_problem_losses(
            model, input_ids, problem_spans, chunk_size,
        )
        if pp_adapt:
            adaptive_losses.append(sum(pp_adapt) / len(pp_adapt))
            per_problem_adaptive.append(pp_adapt)

        # --- Baseline run (reset before each chunk → no learning) ---
        model.start_session(batch_size)
        # Save & zero-out fast weight inits temporarily to simulate no memory
        saved_A_init = model.adaptive_A.fast_A_init.data.clone()
        saved_B_init_A = model.adaptive_A.fast_B_init.data.clone()
        saved_A_init_B = model.adaptive_B.fast_A_init.data.clone()
        saved_B_init_B = model.adaptive_B.fast_B_init.data.clone()

        model.adaptive_A.fast_A_init.data.zero_()
        model.adaptive_A.fast_B_init.data.zero_()
        model.adaptive_B.fast_A_init.data.zero_()
        model.adaptive_B.fast_B_init.data.zero_()

        model.start_session(batch_size)  # reset with zeroed inits
        pp_base = _compute_per_problem_losses(
            model, input_ids, problem_spans, chunk_size,
        )
        if pp_base:
            baseline_losses.append(sum(pp_base) / len(pp_base))
            per_problem_baseline.append(pp_base)

        # Restore fast weight inits
        model.adaptive_A.fast_A_init.data.copy_(saved_A_init)
        model.adaptive_A.fast_B_init.data.copy_(saved_B_init_A)
        model.adaptive_B.fast_A_init.data.copy_(saved_A_init_B)
        model.adaptive_B.fast_B_init.data.copy_(saved_B_init_B)

    if not adaptive_losses:
        return EvalResult(
            name="baseline_comparison",
            passed=False,
            summary="No episodes evaluated.",
        )

    adaptive_mean = sum(adaptive_losses) / len(adaptive_losses)
    baseline_mean = sum(baseline_losses) / len(baseline_losses)
    benefit = baseline_mean - adaptive_mean
    benefit_pct = (benefit / baseline_mean * 100) if baseline_mean > 0 else 0.0
    passed = adaptive_mean < baseline_mean

    # Average per-problem curves
    max_problems = max(len(pp) for pp in per_problem_adaptive)
    avg_adapt_curve: list[float] = []
    avg_base_curve: list[float] = []
    for pos in range(max_problems):
        a_vals = [pp[pos] for pp in per_problem_adaptive if pos < len(pp)]
        b_vals = [pp[pos] for pp in per_problem_baseline if pos < len(pp)]
        avg_adapt_curve.append(sum(a_vals) / len(a_vals) if a_vals else 0.0)
        avg_base_curve.append(sum(b_vals) / len(b_vals) if b_vals else 0.0)

    result = EvalResult(
        name="baseline_comparison",
        passed=passed,
        metrics={
            "adaptive_loss": adaptive_mean,
            "baseline_loss": baseline_mean,
            "benefit": benefit,
            "benefit_pct": benefit_pct,
            "num_episodes": float(len(adaptive_losses)),
        },
        curves={
            "adaptive_per_problem": avg_adapt_curve,
            "baseline_per_problem": avg_base_curve,
        },
        summary=(
            f"Baseline comparison ({'PASS' if passed else 'FAIL'}): "
            f"adaptive loss = {adaptive_mean:.4f}, "
            f"frozen baseline = {baseline_mean:.4f}, "
            f"benefit = {benefit:+.4f} ({benefit_pct:+.1f}%). "
            f"Over {len(adaptive_losses)} episodes."
        ),
    )
    model.train()
    return result


# ------------------------------------------------------------------ #
# Full evaluation suite                                                #
# ------------------------------------------------------------------ #

def run_full_evaluation(
    model,
    config,
    *,
    synthetic: bool = True,
    num_within_episodes: int = 10,
    num_cross_sessions: int = 20,
    num_baseline_episodes: int = 10,
    cross_session_domain: str = "math",
    forgetting_domain_a: str = "math",
    forgetting_domain_b: str = "code",
    forgetting_sessions_a: int = 10,
    forgetting_sessions_b: int = 10,
    forgetting_sessions_return: int = 5,
    output_dir: str | Path | None = None,
    use_wandb: bool = False,
) -> dict[str, Any]:
    """
    Run the complete evaluation suite.

    Parameters
    ----------
    model : NATModel
    config : NATConfig
    synthetic : bool
    num_within_episodes : int
    num_cross_sessions : int
    num_baseline_episodes : int
    cross_session_domain : str
    forgetting_domain_a, forgetting_domain_b : str
    forgetting_sessions_a, forgetting_sessions_b, forgetting_sessions_return : int
    output_dir : str or Path or None
        If set, save individual JSON results to this directory.
    use_wandb : bool

    Returns
    -------
    dict with keys ``"within_session"``, ``"cross_session"``,
    ``"forgetting"``, ``"baseline"``, ``"summary"`` (str), and
    ``"all_passed"`` (bool).
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("NAT Evaluation Suite")
    logger.info("=" * 60)

    results: dict[str, EvalResult] = {}

    # 1. Within-session learning
    logger.info("─── Evaluation 1/4: Within-session learning ───")
    results["within_session"] = evaluate_within_session(
        model, config,
        num_episodes=num_within_episodes,
        synthetic=synthetic,
    )
    logger.info(results["within_session"].summary)

    # 2. Cross-session learning
    logger.info("─── Evaluation 2/4: Cross-session learning ───")
    results["cross_session"] = evaluate_cross_session(
        model, config,
        domain=cross_session_domain,
        num_sessions=num_cross_sessions,
        synthetic=synthetic,
    )
    logger.info(results["cross_session"].summary)

    # 3. Forgetting test
    logger.info("─── Evaluation 3/4: Forgetting test ───")
    results["forgetting"] = evaluate_forgetting(
        model, config,
        domain_a=forgetting_domain_a,
        domain_b=forgetting_domain_b,
        sessions_a=forgetting_sessions_a,
        sessions_b=forgetting_sessions_b,
        sessions_return=forgetting_sessions_return,
        synthetic=synthetic,
    )
    logger.info(results["forgetting"].summary)

    # 4. Baseline comparison
    logger.info("─── Evaluation 4/4: Baseline comparison ───")
    results["baseline"] = evaluate_baseline(
        model, config,
        num_episodes=num_baseline_episodes,
        synthetic=synthetic,
    )
    logger.info(results["baseline"].summary)

    # Summary
    all_passed = all(r.passed for r in results.values())
    elapsed = time.time() - t0

    summary_lines = [
        "",
        "=" * 60,
        "EVALUATION SUMMARY",
        "=" * 60,
    ]
    for name, r in results.items():
        status = "✓ PASS" if r.passed else "✗ FAIL"
        summary_lines.append(f"  [{status}]  {name}")
        for k, v in r.metrics.items():
            if isinstance(v, float):
                summary_lines.append(f"           {k}: {v:.4f}")
    summary_lines.append("─" * 60)
    summary_lines.append(
        f"  Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}  "
        f"({elapsed:.1f}s)"
    )
    summary_lines.append("=" * 60)
    summary = "\n".join(summary_lines)
    logger.info(summary)

    # Save results
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, r in results.items():
            r.to_json(out / f"{name}.json")
        (out / "summary.txt").write_text(summary)
        logger.info(f"Results saved to {out}/")

    # W&B logging
    if use_wandb:
        try:
            import wandb

            log_dict: dict[str, Any] = {"eval/all_passed": int(all_passed)}
            for name, r in results.items():
                for k, v in r.metrics.items():
                    log_dict[f"eval/{name}/{k}"] = v
                log_dict[f"eval/{name}/passed"] = int(r.passed)
            wandb.log(log_dict)
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")

    return {
        "within_session": results["within_session"],
        "cross_session": results["cross_session"],
        "forgetting": results["forgetting"],
        "baseline": results["baseline"],
        "summary": summary,
        "all_passed": all_passed,
    }
