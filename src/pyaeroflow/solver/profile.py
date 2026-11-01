"""Wall-time breakdown of the implicit iteration (accumulated per run).

``with timers.section("factorize"):`` adds the elapsed time and one call to the
named section. Sections may nest (``precondition`` and ``jvp`` run inside
``gmres``), so the percentages in the summary refer to the wall time of the run
and do not add up to 100 %.
"""

import time
from contextlib import contextmanager


class Timers:
    """Accumulated seconds and call counts per named section."""

    def __init__(self):
        self.seconds = {}
        self.calls = {}

    def reset(self):
        """Forget all sections."""
        self.seconds.clear()
        self.calls.clear()

    @contextmanager
    def section(self, name):
        """Time the body of a ``with`` block under ``name``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t0
            self.calls[name] = self.calls.get(name, 0) + 1

    def rows(self, wall):
        """Return ``(label, text)`` rows ordered by time, for the run log."""
        out = []
        for name, sec in sorted(self.seconds.items(), key=lambda kv: -kv[1]):
            n = self.calls[name]
            out.append(
                (
                    name,
                    f"{sec:.1f} s ({100 * sec / max(wall, 1e-12):.0f} %), "
                    f"{n} calls, {sec / n:.2f} s each",
                )
            )
        return out


timers = Timers()
