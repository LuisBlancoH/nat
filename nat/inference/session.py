"""
Session management for NAT inference.

A *session* is the fundamental unit of inference-time learning.  Within a
session the adaptive layers accumulate knowledge in their fast weights;
between sessions the consolidation layer absorbs that knowledge via EMA
and the fast weights are partially reset.

This module provides :class:`SessionManager` — a high-level wrapper around
:class:`NATModel` that handles:

* Loading a trained checkpoint (θ) and optional consolidated state.
* Starting / ending sessions with proper lifecycle calls.
* Saving / loading consolidated memory across process restarts.
* Multi-session workflows (e.g. "read five documents, then query").
* Diagnostic logging of fast-weight and consolidation statistics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch

from nat.config import NATConfig
from nat.model.nat_model import NATModel
from nat.training.phase1_meta_learn import load_checkpoint

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Session metadata                                                     #
# ------------------------------------------------------------------ #

@dataclass
class SessionInfo:
    """Lightweight record of a completed session."""

    session_id: int
    tokens_processed: int
    adaptation_steps: int
    fast_A_norm_start: float
    fast_A_norm_end: float
    fast_B_norm_start: float
    fast_B_norm_end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ #
# SessionManager                                                       #
# ------------------------------------------------------------------ #

class SessionManager:
    """
    High-level session controller for NAT inference.

    Typical usage::

        mgr = SessionManager.from_checkpoint(
            "configs/apple_silicon.yaml",
            "checkpoints/phase3.pt",
        )

        # --- Session 1: feed some context ---
        mgr.start_session()
        mgr.feed("The capital of France is Paris. ...")
        mgr.end_session()          # consolidation happens here

        # --- Session 2: the model now "remembers" ---
        mgr.start_session()
        text = mgr.generate("What is the capital of France?")
        mgr.end_session()

        # Persist consolidated memory to disk
        mgr.save_consolidated("memory/consolidated.pt")

    Parameters
    ----------
    model : NATModel
        A fully initialised (and preferably checkpoint-loaded) model.
    config : NATConfig
        Configuration used to build the model.
    """

    def __init__(self, model: NATModel, config: NATConfig):
        self.model = model
        self.config = config

        self._device = torch.device(config.device)
        self.model.to(self._device)
        self.model.eval()

        self._session_active = False
        self._session_count = 0
        self._history: list[SessionInfo] = []
        self._tokens_this_session = 0

    # ------------------------------------------------------------------ #
    # Construction helpers                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
        consolidated_path: str | Path | None = None,
        device: str | None = None,
    ) -> "SessionManager":
        """
        Build a ready-to-use SessionManager from disk artefacts.

        Parameters
        ----------
        config_path : str | Path
            Path to a YAML config file.
        checkpoint_path : str | Path | None
            Path to a Phase-1/2/3 checkpoint containing trained θ.
            If ``None``, the model starts with untrained θ.
        consolidated_path : str | Path | None
            Path to a previously saved consolidated state.
            If ``None``, consolidated weights start at zero.
        device : str | None
            Override the config's device setting.

        Returns
        -------
        SessionManager
        """
        config = NATConfig.from_yaml(str(config_path))
        if device is not None:
            config.device = device

        logger.info(f"Building NATModel from {config.base_model_name} …")
        model = NATModel(config)

        if checkpoint_path is not None:
            load_checkpoint(model, str(checkpoint_path))

        if consolidated_path is not None:
            model.consolidation.load_state(str(consolidated_path))
            logger.info(f"Loaded consolidated state from {consolidated_path}")

        return cls(model, config)

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def start_session(self, batch_size: int = 1) -> None:
        """
        Begin a new session.

        Resets fast weights (with partial retention from consolidation)
        and prepares the model for a new stream of input.

        Raises
        ------
        RuntimeError
            If a session is already active (call ``end_session`` first).
        """
        if self._session_active:
            raise RuntimeError(
                "A session is already active.  Call end_session() first."
            )
        self._session_count += 1
        self.model.start_session(batch_size)
        self._session_active = True
        self._tokens_this_session = 0

        # Snapshot fast-weight norms at session start
        stats = self.model.diagnostics()
        self._start_stats = stats
        logger.debug(f"Session {self._session_count} started.")

    def end_session(self) -> SessionInfo:
        """
        End the current session.

        Triggers consolidation (EMA update of consolidated weights from
        current fast weights) and partial reset.

        Returns
        -------
        SessionInfo
            A summary of the session that just ended.

        Raises
        ------
        RuntimeError
            If no session is active.
        """
        if not self._session_active:
            raise RuntimeError("No active session to end.")

        end_stats = self.model.diagnostics()

        self.model.end_session()
        self._session_active = False

        info = SessionInfo(
            session_id=self._session_count,
            tokens_processed=self._tokens_this_session,
            adaptation_steps=self._tokens_this_session // self.model.adapt_every_n,
            fast_A_norm_start=self._start_stats.get("adaptive_A/fast_A_norm", 0.0),
            fast_A_norm_end=end_stats.get("adaptive_A/fast_A_norm", 0.0),
            fast_B_norm_start=self._start_stats.get("adaptive_A/fast_B_norm", 0.0),
            fast_B_norm_end=end_stats.get("adaptive_A/fast_B_norm", 0.0),
        )
        self._history.append(info)
        logger.info(
            f"Session {info.session_id} ended — "
            f"{info.tokens_processed} tokens, "
            f"{info.adaptation_steps} adaptations, "
            f"‖W_A‖ {info.fast_A_norm_start:.3f}→{info.fast_A_norm_end:.3f}"
        )
        return info

    @property
    def session_active(self) -> bool:
        return self._session_active

    @property
    def session_count(self) -> int:
        return self._session_count

    @property
    def history(self) -> list[SessionInfo]:
        return list(self._history)

    # ------------------------------------------------------------------ #
    # Feed / process text                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def feed(
        self,
        text: str,
        *,
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Feed text into the model so the adaptive layers can learn from it.

        This is the "study" / "reading" phase — no text is generated.
        The model processes the text and adapts its fast weights.

        Parameters
        ----------
        text : str
            The text to process.
        chunk_size : int | None
            Process the text in chunks of this many tokens.
            Defaults to ``config.seq_len``.

        Returns
        -------
        dict
            ``{"tokens_processed": int, "chunks": int, "diagnostics": dict}``

        Raises
        ------
        RuntimeError
            If no session is active.
        """
        if not self._session_active:
            raise RuntimeError("No active session.  Call start_session() first.")

        tokenizer = self.model.tokenizer
        if tokenizer is None:
            raise RuntimeError("Model has no tokenizer — cannot feed text.")

        chunk_size = chunk_size or self.config.seq_len
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].to(self._device)
        total_tokens = input_ids.shape[1]

        num_chunks = 0
        for start in range(0, total_tokens, chunk_size):
            chunk = input_ids[:, start : start + chunk_size]
            self.model(chunk)
            num_chunks += 1

        self._tokens_this_session += total_tokens
        diag = self.model.diagnostics()

        logger.debug(
            f"Fed {total_tokens} tokens in {num_chunks} chunks."
        )
        return {
            "tokens_processed": total_tokens,
            "chunks": num_chunks,
            "diagnostics": diag,
        }

    @torch.no_grad()
    def feed_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Feed pre-tokenised input (avoids re-tokenising).

        Parameters
        ----------
        input_ids : LongTensor, shape ``(1, seq_len)`` or ``(seq_len,)``
        chunk_size : int | None
        """
        if not self._session_active:
            raise RuntimeError("No active session.  Call start_session() first.")

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self._device)
        chunk_size = chunk_size or self.config.seq_len
        total_tokens = input_ids.shape[1]

        num_chunks = 0
        for start in range(0, total_tokens, chunk_size):
            chunk = input_ids[:, start : start + chunk_size]
            self.model(chunk)
            num_chunks += 1

        self._tokens_this_session += total_tokens
        return {
            "tokens_processed": total_tokens,
            "chunks": num_chunks,
            "diagnostics": self.model.diagnostics(),
        }

    # ------------------------------------------------------------------ #
    # Persistence — consolidated memory                                    #
    # ------------------------------------------------------------------ #

    def save_consolidated(self, path: str | Path) -> Path:
        """
        Save the consolidated memory to disk.

        This allows resuming inference across process restarts without
        losing cross-session knowledge.

        Parameters
        ----------
        path : str | Path
            Destination file path.

        Returns
        -------
        Path
            The resolved path that was written.
        """
        path = Path(path)
        self.model.consolidation.save_state(path)
        logger.info(f"Consolidated state saved → {path}")
        return path

    def load_consolidated(self, path: str | Path) -> None:
        """
        Load consolidated memory from disk.

        Parameters
        ----------
        path : str | Path
            Path to a previously saved consolidated state.
        """
        self.model.consolidation.load_state(str(path))
        logger.info(f"Consolidated state loaded ← {path}")

    # ------------------------------------------------------------------ #
    # Persistence — full session history                                   #
    # ------------------------------------------------------------------ #

    def save_history(self, path: str | Path) -> Path:
        """Save session history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_count": self._session_count,
            "sessions": [s.to_dict() for s in self._history],
            "consolidation_stats": self.model.consolidation.consolidated_weight_stats(),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Session history saved → {path}")
        return path

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def diagnostics(self) -> dict[str, Any]:
        """Return current model diagnostics."""
        d = self.model.diagnostics()
        d["session_count"] = self._session_count
        d["session_active"] = self._session_active
        d["tokens_this_session"] = self._tokens_this_session
        d["consolidation_empty"] = self.model.consolidation.is_empty
        return d

    def summary(self) -> str:
        """Return a human-readable summary string."""
        d = self.diagnostics()
        lines = [
            f"NAT SessionManager",
            f"  Sessions completed: {d['session_count']}",
            f"  Session active:     {d['session_active']}",
            f"  Consolidation empty:{d['consolidation_empty']}",
        ]
        if d["session_active"]:
            lines.append(f"  Tokens this session:{d['tokens_this_session']}")
        if self._history:
            last = self._history[-1]
            lines.append(
                f"  Last session:       #{last.session_id} "
                f"({last.tokens_processed} tokens)"
            )
        return "\n".join(lines)
