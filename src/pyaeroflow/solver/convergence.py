"""Force-based convergence of a steady run (SU2's Cauchy criterion)."""

import numpy as np


class CauchyCriterion:
    """Cauchy convergence of monitored coefficients, as SU2's CONV_FIELD.

    Every accepted iteration adds the change of each coefficient to a window
    of the last ``elements`` changes. The criterion is met when the mean
    absolute change of every monitored coefficient is below ``eps`` and the
    window is full, from iteration ``start`` on.

    Parameters
    ----------
    names : sequence of str
        Monitored coefficients, ``cl``, ``cd`` or ``cm``.
    elements : int
        Window length in iterations (SU2 CONV_CAUCHY_ELEMS).
    eps : float
        Tolerance of the mean absolute change per iteration (CONV_CAUCHY_EPS).
    start : int
        Iteration the criterion counts from (CONV_STARTITER).
    """

    ORDER = ("cl", "cd", "cm")

    def __init__(self, names, elements=100, eps=1.0e-10, start=0):
        self.names = tuple(names)
        self.index = [self.ORDER.index(n) for n in self.names]
        self.elements, self.eps, self.start = elements, eps, start
        self.last = None
        self.changes = []

    def update(self, iteration, coefficients):
        """Add the coefficients ``(CL, CD, CM)`` of an accepted iteration.

        Returns True once the criterion is met.
        """
        values = np.array([coefficients[i] for i in self.index], dtype=float)
        if self.last is not None and iteration >= self.start:
            self.changes.append(np.abs(values - self.last))
            del self.changes[: -self.elements]
        self.last = values
        return self.converged

    @property
    def value(self):
        """Mean absolute change per coefficient over the window (None if empty)."""
        if not self.changes:
            return None
        return np.mean(self.changes, axis=0)

    @property
    def converged(self):
        """True if the window is full and every mean change is below ``eps``."""
        return len(self.changes) >= self.elements and bool(
            np.all(self.value < self.eps)
        )
