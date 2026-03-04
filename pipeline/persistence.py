from __future__ import annotations

from typing import Callable

from .types import PersistenceInputs


def persist_pipeline_outputs(
    *,
    inputs: PersistenceInputs,
    persist_fn: Callable[[dict], None],
) -> None:
    # Persistence is intentionally delegated to the existing writer to preserve exact output semantics.
    persist_fn(inputs.snapshot)
