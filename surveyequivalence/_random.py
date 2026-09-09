"""Scoped random streams for reproducible, independently scheduled tasks."""

from contextlib import contextmanager
import random

import numpy as np


@contextmanager
def random_stream(seed, *task):
    """Seed legacy Python/NumPy consumers without changing caller RNG state.

    Integer task coordinates are stable logical identifiers, never worker IDs.
    A missing seed retains the historical use of the caller's random streams.
    """
    if seed is None:
        yield
        return
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    words = np.random.SeedSequence([int(seed), *map(int, task)]).generate_state(4)
    try:
        random.seed(int.from_bytes(words.tobytes(), "little"))
        np.random.seed(words)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
