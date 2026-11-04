"""Force-based (Cauchy) convergence of steady and unsteady runs.

Steady: :class:`CauchyCriterion` on the relative change of the coefficients
between iterations. Unsteady: :class:`WindowedCauchyCriterion` on the windowed
time average of the coefficients (window functions after Krakos et al. 2012).
"""

import numpy as np

TINY = np.finfo(float).tiny


class CauchyCriterion:
    """Cauchy convergence of monitored coefficients.

    A Cauchy element is the relative difference of a coefficient between two
    consecutive iterations, ``|C_n - C_(n-1)| / |C_n|`` (a
    coefficient near zero, e.g. C_L at alpha = 0, gives large elements).
    Every accepted iteration adds the element of each coefficient to a window
    of the last ``elements`` elements. The criterion is met when the mean
    of every monitored coefficient is below ``eps`` and the window is full,
    from iteration ``start`` on.

    Forces that stand still do not prove convergence: a run whose updates are
    cut back by the line search at every iteration can freeze its forces while
    its residual sits at the peak. With ``min_drop`` the window must also be
    made of iterations whose residual drop (the worst equation relative to its
    own peak, in orders) is at least ``min_drop``; an iteration below it
    starts the count again. ``min_drop = 0`` is the plain Cauchy criterion.

    Parameters
    ----------
    names : sequence of str
        Monitored coefficients, ``cl``, ``cd`` or ``cm``.
    elements : int
        Window length in iterations.
    eps : float
        Tolerance of the mean absolute change per iteration.
    start : int
        Iteration the criterion counts from.
    min_drop : float
        Residual drop in orders that every iteration of the window needs;
        ``update`` is given the drop of each iteration (None = not known, the
        iteration counts).
    """

    ORDER = ("cl", "cd", "cm")

    def __init__(self, names, elements=100, eps=1.0e-10, start=0, min_drop=0.0):
        self.names = tuple(names)
        self.index = [self.ORDER.index(n) for n in self.names]
        self.elements, self.eps, self.start = elements, eps, start
        self.min_drop = min_drop
        self.last = None
        self.changes = []
        self.settled = 0  # trailing window elements with a residual drop >= min_drop

    def update(self, iteration, coefficients, drop=None):
        """Add the coefficients ``(CL, CD, CM)`` of an accepted iteration.

        ``drop`` is the residual drop of the iteration in orders (the ``drop``
        column of the log). Returns True once the criterion is met.
        """
        values = np.array([coefficients[i] for i in self.index], dtype=float)
        if self.last is not None and iteration >= self.start:
            self.changes.append(
                np.abs(values - self.last) / np.maximum(np.abs(values), TINY)
            )
            del self.changes[: -self.elements]
            if drop is None or drop >= self.min_drop:
                self.settled = min(self.settled + 1, self.elements)
            else:
                self.settled = 0
        self.last = values
        return self.converged

    @property
    def value(self):
        """Mean relative change per coefficient over the window (None if empty)."""
        if not self.changes:
            return None
        return np.mean(self.changes, axis=0)

    @property
    def steady(self):
        """True if the window is full and every mean change is below ``eps``.

        This is the plain criterion, whatever the residual does.
        """
        return len(self.changes) >= self.elements and bool(
            np.all(self.value < self.eps)
        )

    @property
    def held_back(self):
        """True if the forces are steady but the residual drop is too small."""
        return self.steady and self.settled < self.elements

    @property
    def converged(self):
        """True if the forces are steady and the residual has dropped enough.

        The drop must be ``min_drop`` orders or more over the whole window.
        """
        return self.steady and self.settled >= self.elements


WINDOW_FUNCTIONS = ("square", "hann", "hann_square", "bump")


def window_weight(function, i, n):
    """Weight of sample ``i`` (0 .. n-1) of a window of ``n`` samples.

    ``square`` is 1 (plain average), ``hann`` is 0.5 (1 - cos(2 pi i / (n-1))),
    ``hann_square`` its square, ``bump`` exp(2 exp(-1 / (1 + x)) / (x - 1)) with
    x = 2 i / (n-1) - 1; all but ``square`` vanish at both ends.
    """
    if function == "square" or n < 2:
        return 1.0
    t = i / (n - 1)
    if function == "hann":
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
    if function == "hann_square":
        return (0.5 * (1.0 - np.cos(2.0 * np.pi * t))) ** 2
    if function == "bump":
        x = 2.0 * t - 1.0
        if abs(x) >= 1.0:
            return 0.0
        return float(np.exp(2.0 * np.exp(-1.0 / (1.0 + x)) / (x - 1.0)))
    raise ValueError(f"window function must be one of {WINDOW_FUNCTIONS}")


class WindowedCauchyCriterion:
    """Cauchy convergence of the windowed time average of coefficients.

    From time step ``window_start`` on, each
    monitored coefficient is averaged over the steps with the weights of a
    window function over the ``n_window`` steps from ``window_start`` to the
    last step of the run (``tavg`` is the weighted mean sum(w C) / sum(w)).
    A Cauchy element is the relative change of this average between two
    consecutive steps. From step ``window_start + start`` on the elements
    enter a window of the last ``elements``; the criterion is met when the
    window is full and the mean element of every coefficient is below ``eps``.

    Parameters
    ----------
    names : sequence of str
        Monitored coefficients, ``cl``, ``cd`` or ``cm``.
    n_window : int
        Number of steps in the averaging window.
    function : str
        One of :data:`WINDOW_FUNCTIONS`.
    elements, eps, start, window_start : as described above.
    """

    ORDER = CauchyCriterion.ORDER

    def __init__(
        self,
        names,
        n_window,
        function="square",
        elements=100,
        eps=1.0e-3,
        start=0,
        window_start=0,
    ):
        self.names = tuple(names)
        self.index = [self.ORDER.index(n) for n in self.names]
        self.n_window, self.function = max(int(n_window), 1), function
        self.elements, self.eps = elements, eps
        self.start, self.window_start = start, window_start
        self.sum_w = 0.0
        self.sum_wc = np.zeros(len(self.names))
        self.average = None
        self.changes = []

    def update(self, step, coefficients):
        """Add the coefficients ``(CL, CD, CM)`` of time step ``step``.

        Returns True once the criterion is met.
        """
        if step < self.window_start:
            return False
        values = np.array([coefficients[i] for i in self.index], dtype=float)
        w = window_weight(self.function, step - self.window_start, self.n_window)
        self.sum_w += w
        self.sum_wc += w * values
        if self.sum_w <= 0.0:
            return False
        average = self.sum_wc / self.sum_w
        if self.average is not None and step >= self.window_start + self.start:
            self.changes.append(
                np.abs(average - self.average) / np.maximum(np.abs(average), TINY)
            )
            del self.changes[: -self.elements]
        self.average = average
        return self.converged

    @property
    def value(self):
        """Mean relative change per coefficient over the window (None if empty)."""
        if not self.changes:
            return None
        return np.mean(self.changes, axis=0)

    @property
    def converged(self):
        """True if the window is full and every mean change is below ``eps``."""
        return len(self.changes) >= self.elements and bool(
            np.all(self.value < self.eps)
        )
