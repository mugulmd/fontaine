"""Per-item randomness.

Every stream item derives its own generator from ``(seed, index)`` rather than
drawing from one shared sequence. Item 12403 is then reproducible on its own,
without replaying the 12402 items before it — which is what makes a failure deep
in a long stream debuggable.
"""

from __future__ import annotations

import random


def item_rng(seed: int, index: int, *, purpose: str = "") -> random.Random:
    """A generator dedicated to one stream item, seeded reproducibly.

    ``random.Random`` hashes string seeds with SHA-512, so this is stable across
    processes and platforms — unlike ``hash()``, which is not.
    """
    return random.Random(f"fontaine:{seed}:{index}:{purpose}")
