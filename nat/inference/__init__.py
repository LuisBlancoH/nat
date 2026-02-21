"""
NAT Inference — session management and text generation.

Public API
----------
.. autoclass:: SessionManager
.. autoclass:: SessionInfo
.. autoclass:: GenerationConfig
.. autoclass:: GenerationResult
.. autofunction:: generate
.. autofunction:: generate_text
.. autofunction:: generate_with_context
"""

from nat.inference.session import SessionManager, SessionInfo
from nat.inference.generate import (
    GenerationConfig,
    GenerationResult,
    generate,
    generate_text,
    generate_with_context,
)

__all__ = [
    "SessionManager",
    "SessionInfo",
    "GenerationConfig",
    "GenerationResult",
    "generate",
    "generate_text",
    "generate_with_context",
]
