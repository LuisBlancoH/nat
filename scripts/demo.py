#!/usr/bin/env python
"""
Interactive demo of NAT inference-time learning.

Demonstrates the core NAT capability: the model reads a context document,
its adaptive layers learn from it in real time, and subsequent generations
benefit from that self-acquired knowledge.

Usage
-----
::

    # Interactive mode with a trained checkpoint
    python scripts/demo.py --config configs/apple_silicon.yaml \\
        --checkpoint checkpoints/phase3.pt

    # One-shot: feed context and generate
    python scripts/demo.py --config configs/apple_silicon.yaml \\
        --checkpoint checkpoints/phase3.pt \\
        --context "The Eiffel Tower was built in 1889 for the World Fair." \\
        --prompt "When was the Eiffel Tower built?"

    # Without checkpoint (untrained θ — useful for testing the pipeline)
    python scripts/demo.py --config configs/small.yaml

    # Load prior consolidated memory
    python scripts/demo.py --config configs/apple_silicon.yaml \\
        --checkpoint checkpoints/phase3.pt \\
        --consolidated memory/consolidated.pt
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from nat.config import NATConfig
from nat.model.nat_model import NATModel
from nat.training.phase1_meta_learn import load_checkpoint
from nat.inference.session import SessionManager
from nat.inference.generate import (
    GenerationConfig,
    GenerationResult,
    generate,
    generate_text,
    generate_with_context,
)


def build_model(args) -> tuple[NATModel, "transformers.PreTrainedTokenizer"]:
    """Build and load the model + tokenizer."""
    config = NATConfig.from_yaml(args.config)
    if args.device:
        config.device = args.device

    logging.info(f"Loading base model: {config.base_model_name}")
    model = NATModel(config)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
        logging.info(f"Loaded checkpoint: {args.checkpoint}")

    if args.consolidated:
        model.consolidation.load_state(args.consolidated)
        logging.info(f"Loaded consolidated state: {args.consolidated}")

    device = torch.device(config.device)
    model.to(device)
    model.eval()

    return model, model.tokenizer


def one_shot_demo(model, tokenizer, args):
    """Feed context, generate, print result."""
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    if args.context:
        print(f"\n📖 Context: {args.context[:200]}{'...' if len(args.context) > 200 else ''}")
        print(f"❓ Prompt:  {args.prompt}")
        print("─" * 60)

        result = generate_with_context(
            model,
            context=args.context,
            prompt=args.prompt,
            tokenizer=tokenizer,
            gen_config=gen_config,
        )
    else:
        print(f"❓ Prompt: {args.prompt}")
        print("─" * 60)

        device = next(model.parameters()).device
        model.start_session(1)
        input_ids = tokenizer(
            args.prompt, return_tensors="pt", add_special_tokens=True,
        )["input_ids"].to(device)

        result = generate(
            model,
            input_ids,
            gen_config=gen_config,
            tokenizer=tokenizer,
        )

    print(f"🤖 Response: {result.text}")
    print("─" * 60)
    print(f"   Tokens generated: {result.num_tokens_generated}")
    print(f"   Stop reason:      {result.stop_reason}")
    print(f"   Adaptation steps: {result.adaptation_steps}")


def interactive_demo(model, tokenizer, args):
    """Multi-session interactive REPL."""
    mgr = SessionManager(model, NATConfig.from_yaml(args.config))

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        NAT Interactive Demo — Inference-Time Learning   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Commands:                                              ║")
    print("║    /session     — start a new session                   ║")
    print("║    /end         — end current session (consolidate)     ║")
    print("║    /feed <text> — feed text into current session        ║")
    print("║    /status      — show session status                   ║")
    print("║    /save <path> — save consolidated memory              ║")
    print("║    /load <path> — load consolidated memory              ║")
    print("║    /quit        — exit                                  ║")
    print("║                                                         ║")
    print("║  Any other input → generate a response                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        elif user_input.lower() == "/session":
            if mgr.session_active:
                print("⚠  Ending current session first...")
                info = mgr.end_session()
                print(f"   Session {info.session_id} ended.")
            mgr.start_session()
            print(f"✓ Session {mgr.session_count} started.")

        elif user_input.lower() == "/end":
            if not mgr.session_active:
                print("⚠  No active session.")
                continue
            info = mgr.end_session()
            print(f"✓ Session {info.session_id} ended — "
                  f"{info.tokens_processed} tokens processed.")

        elif user_input.lower().startswith("/feed "):
            text = user_input[6:]
            if not mgr.session_active:
                mgr.start_session()
                print(f"✓ Session {mgr.session_count} auto-started.")
            result = mgr.feed(text)
            print(f"✓ Fed {result['tokens_processed']} tokens.")

        elif user_input.lower() == "/status":
            print(mgr.summary())
            diag = mgr.diagnostics()
            c_stats = model.consolidation.consolidated_weight_stats()
            print(f"  ‖W_c_A‖ = {c_stats['W_c_A_norm']:.4f}")
            print(f"  ‖W_c_B‖ = {c_stats['W_c_B_norm']:.4f}")

        elif user_input.lower().startswith("/save "):
            path = user_input[6:].strip()
            mgr.save_consolidated(path)
            print(f"✓ Saved consolidated memory to {path}")

        elif user_input.lower().startswith("/load "):
            path = user_input[6:].strip()
            try:
                mgr.load_consolidated(path)
                print(f"✓ Loaded consolidated memory from {path}")
            except FileNotFoundError:
                print(f"✗ File not found: {path}")

        else:
            # Generate a response
            if not mgr.session_active:
                mgr.start_session()
                print(f"  (auto-started session {mgr.session_count})")

            device = next(model.parameters()).device
            input_ids = tokenizer(
                user_input, return_tensors="pt", add_special_tokens=True,
            )["input_ids"].to(device)

            result = generate(
                model,
                input_ids,
                gen_config=gen_config,
                tokenizer=tokenizer,
            )
            print(f"NAT> {result.text}")
            print(f"     [{result.num_tokens_generated} tokens, "
                  f"{result.adaptation_steps} adaptations, "
                  f"stop={result.stop_reason}]")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="NAT Interactive Demo — Inference-Time Learning"
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained checkpoint (phase1/2/3)",
    )
    parser.add_argument(
        "--consolidated", type=str, default=None,
        help="Path to consolidated memory state",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda, mps, cpu)",
    )

    # One-shot mode
    parser.add_argument(
        "--context", type=str, default=None,
        help="Context text for the model to learn from (one-shot mode)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Prompt to generate from (one-shot mode)",
    )

    # Generation params
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    model, tokenizer = build_model(args)

    if tokenizer is None:
        print("ERROR: Model has no tokenizer.  Cannot run demo.")
        sys.exit(1)

    if args.prompt:
        one_shot_demo(model, tokenizer, args)
    else:
        interactive_demo(model, tokenizer, args)


if __name__ == "__main__":
    main()
