"""Refinement indicators and marking for local refinement (R3).

An indicator is an arithmetic expression over named cell fields, for example

    pressure_gradient_magnitude * h
    pressure_gradient_magnitude * sqrt(2/3 * k) / velocity_magnitude
    norm(vorticity) * h**2 * (wall_distance > 1e-4)

evaluated per cell by a small whitelisted parser (no ``eval``): numbers,
``+ - * / **``, comparisons (giving 0 or 1), parentheses, names, and the
functions in :data:`FUNCTIONS`. Names are the output fields of the solver
(:mod:`pyaeroflow.io.fields`) plus the geometric variables of
:func:`geometry_variables`: ``x``, ``y``, ``area``, ``h`` (square root of the
area), ``h_u``, ``h_v`` (mean side lengths along the two directions of a
quad), ``aspect`` (long over short side) and ``factor_u``, ``factor_v``
(accumulated split factors). Vector fields enter through ``norm(v)``.

:class:`RefineSpec` says which cells to mark (threshold on the indicator, a
quantile, or a top fraction, with caps) and how to split them;
:func:`plan` turns an indicator into split factors and :func:`apply`
executes them on a :class:`~pyaeroflow.mesh.refine.Refinement`.

Splits of a cell with ``aspect > aspect_target`` go along its long side only
(``n = ceil(aspect / aspect_target)``, at most ``n_max``), which lowers the
aspect ratio of thin wall cells and keeps their thickness; other cells are
split 2 x 2. ``direction = "iso"`` splits all cells 2 x 2.
"""

import ast
import difflib
import math
import operator
from dataclasses import dataclass

import numpy as np

GEOMETRY_NAMES = (
    "x",
    "y",
    "area",
    "h",
    "h_u",
    "h_v",
    "aspect",
    "factor_u",
    "factor_v",
)


class IndicatorError(ValueError):
    """Raised for an invalid indicator expression or refinement setting."""


def _norm(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 2:
        return np.hypot(v[:, 0], v[:, 1])
    return np.abs(v)


FUNCTIONS = {
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "tanh": np.tanh,
    "sign": np.sign,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "norm": _norm,
}
CONSTANTS = {"pi": math.pi}

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_PLAIN = (
    ast.Expression,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)
_MAX_LENGTH = 500


def parse(expression):
    """Parse and validate ``expression``; return its syntax tree.

    Raises
    ------
    IndicatorError
        For anything outside the whitelist (attributes, subscripts, lambdas,
        unknown functions, keyword arguments, ...).
    """
    if not isinstance(expression, str) or not expression.strip():
        raise IndicatorError("indicator: the expression is empty")
    if len(expression) > _MAX_LENGTH:
        raise IndicatorError(f"indicator: expression longer than {_MAX_LENGTH} chars")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise IndicatorError(f"indicator: syntax error: {exc.msg}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise IndicatorError(
                    f"indicator: unknown function; allowed: {', '.join(FUNCTIONS)}"
                )
            if node.keywords:
                raise IndicatorError("indicator: keyword arguments are not allowed")
        elif isinstance(node, ast.Compare):
            if any(type(op) not in _COMPARE for op in node.ops) or len(node.ops) != 1:
                raise IndicatorError("indicator: only single < <= > >= comparisons")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise IndicatorError("indicator: allowed operators are + - * / **")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise IndicatorError("indicator: only unary + and -")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise IndicatorError("indicator: only numeric constants")
        elif not isinstance(node, _PLAIN):
            raise IndicatorError(
                f"indicator: '{type(node).__name__}' is not allowed in an expression"
            )
    return tree


def names_used(expression):
    """Variable names in ``expression`` (functions and constants excluded)."""
    tree = parse(expression)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return sorted(names - called - set(CONSTANTS))


def evaluate(expression, variables):
    """Evaluate ``expression`` with ``variables`` (``{name: array}``).

    Returns
    -------
    ndarray, shape (n_cells,)
        The indicator. Vector variables must be wrapped in ``norm()``.
    """
    tree = parse(expression)
    sizes = {np.shape(v)[0] for v in variables.values() if np.ndim(v) > 0}
    if len(sizes) != 1:
        raise IndicatorError("indicator: variables must have one value per cell")
    n = sizes.pop()

    def value(node):
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return np.asarray(variables[node.id], dtype=float)
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            close = difflib.get_close_matches(node.id, list(variables), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            raise IndicatorError(
                f"indicator: unknown name '{node.id}'{hint}; this case provides "
                f"{', '.join(sorted(variables))}"
            )
        if isinstance(node, ast.UnaryOp):
            v = value(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.Compare):
            op = _COMPARE[type(node.ops[0])]
            return op(value(node.left), value(node.comparators[0])).astype(float)
        return FUNCTIONS[node.func.id](*[value(a) for a in node.args])

    try:
        with np.errstate(all="ignore"):
            result = np.asarray(value(tree), dtype=float)
    except (ValueError, TypeError) as exc:
        raise IndicatorError(
            f"indicator: cannot evaluate '{expression}': {exc}; "
            "wrap vector fields in norm()"
        ) from None
    if result.ndim == 0:
        result = np.full(n, float(result))
    if result.shape != (n,):
        raise IndicatorError(
            f"indicator: '{expression}' has shape {result.shape}, expected "
            f"({n},); wrap vector fields in norm()"
        )
    return result


def geometry_variables(ref):
    """Geometric variables of a :class:`~pyaeroflow.mesh.refine.Refinement`."""
    mesh = ref.mesh
    ext = ref.extents()
    factors = ref.split_factors
    return {
        "x": mesh.cell_centroid[:, 0],
        "y": mesh.cell_centroid[:, 1],
        "area": mesh.cell_area,
        "h": np.sqrt(mesh.cell_area),
        "h_u": ext[:, 0],
        "h_v": ext[:, 1],
        "aspect": ext.max(axis=1) / ext.min(axis=1),
        "factor_u": factors[:, 0].astype(float),
        "factor_v": factors[:, 1].astype(float),
    }


@dataclass(frozen=True)
class RefineSpec:
    """What to refine and how.

    Attributes
    ----------
    indicator : str
        Expression, see the module docstring.
    threshold, quantile, top_fraction : float or None
        Exactly one: mark cells with indicator above ``threshold``; above the
        ``quantile`` (0..1) of the indicator over all splittable cells; or the
        ``top_fraction`` (0..1) of the splittable cells with the largest
        indicator.
    max_cells : int or None
        Cap on the cells added by the marked splits (before balancing and the
        boundary-layer column rule add more); the cells with the largest
        indicator are kept.
    max_factor : int
        A cell is not split further along a direction in which its accumulated
        split factor reached this value.
    direction : {"auto", "iso"}
        ``auto`` splits cells with ``aspect > aspect_target`` along their long
        side only, the others 2 x 2; ``iso`` always 2 x 2.
    aspect_target : float
    n_max : int
        Largest split factor of one cell along its long side.
    max_hanging : int or None
    column_aspect : float or None
        Passed to :meth:`~pyaeroflow.mesh.refine.Refinement.split`.
    """

    indicator: str
    threshold: float | None = None
    quantile: float | None = None
    top_fraction: float | None = None
    max_cells: int | None = None
    max_factor: int = 8
    direction: str = "auto"
    aspect_target: float = 3.0
    n_max: int = 4
    max_hanging: int | None = 1
    column_aspect: float | None = 5.0

    def __post_init__(self):
        """Validate the combination of settings."""
        parse(self.indicator)
        given = [
            k
            for k in ("threshold", "quantile", "top_fraction")
            if getattr(self, k) is not None
        ]
        if len(given) != 1:
            raise IndicatorError(
                "refinement: give exactly one of threshold, quantile, top_fraction"
            )
        for key in ("quantile", "top_fraction"):
            v = getattr(self, key)
            if v is not None and not 0.0 < v < 1.0:
                raise IndicatorError(f"refinement: {key} must lie in (0, 1)")
        if self.direction not in ("auto", "iso"):
            raise IndicatorError("refinement: direction must be 'auto' or 'iso'")
        if self.max_factor < 2 or self.n_max < 2:
            raise IndicatorError("refinement: max_factor and n_max must be >= 2")
        if self.aspect_target <= 1.0:
            raise IndicatorError("refinement: aspect_target must exceed 1")
        if self.max_cells is not None and self.max_cells < 1:
            raise IndicatorError("refinement: max_cells must be positive")

    @classmethod
    def from_dict(cls, data):
        """Build from a settings table; unknown keys get a hint."""
        fields = set(cls.__dataclass_fields__)
        for key in data:
            if key not in fields:
                close = difflib.get_close_matches(str(key), sorted(fields), n=1)
                hint = f" (did you mean '{close[0]}'?)" if close else ""
                raise IndicatorError(f"refinement: unknown setting '{key}'{hint}")
        return cls(**data)


@dataclass(frozen=True)
class Marking:
    """Result of :func:`plan`.

    Attributes
    ----------
    cells, nu, nv : ndarray
        Cells to split and their split factors.
    indicator : ndarray
        Indicator of every leaf cell.
    level : float
        Indicator value above which cells were marked.
    n_candidates, n_non_quad, n_nonfinite, n_at_max_factor, n_capped : int
        Cells that could be split, skipped because they are not
        quadrilaterals, skipped for a non-finite indicator, skipped for having
        reached ``max_factor``, and dropped by ``max_cells``.
    """

    cells: np.ndarray
    nu: np.ndarray
    nv: np.ndarray
    indicator: np.ndarray
    level: float
    n_candidates: int
    n_non_quad: int
    n_nonfinite: int
    n_at_max_factor: int
    n_capped: int

    def summary(self):
        """One-line description for the log."""
        return (
            f"marked {self.cells.size} of {self.n_candidates} cells "
            f"(indicator > {self.level:.4g}; skipped: {self.n_non_quad} non-quad, "
            f"{self.n_nonfinite} non-finite, {self.n_at_max_factor} at max factor, "
            f"{self.n_capped} by max_cells)"
        )


def plan(ref, fields, spec):
    """Mark and size the splits for the current leaf cells.

    Parameters
    ----------
    ref : Refinement
    fields : dict of str to ndarray
        Solver fields per leaf cell (``ref.n_cells`` rows).
    spec : RefineSpec

    Returns
    -------
    Marking
    """
    n = ref.n_cells
    variables = geometry_variables(ref)
    for name, v in fields.items():
        if np.shape(v)[0] != n:
            raise IndicatorError(
                f"field '{name}' has {np.shape(v)[0]} values, the mesh has {n} cells"
            )
    variables.update({k: np.asarray(v) for k, v in fields.items()})
    indicator = evaluate(spec.indicator, variables)

    quad = np.array([len(c.corners) == 4 for c in ref.cells])
    finite = np.isfinite(indicator)
    candidates = quad & finite
    ind = np.where(candidates, indicator, -np.inf)

    nu = np.ones(n, dtype=np.int64)
    nv = np.ones(n, dtype=np.int64)
    factors = ref.split_factors
    ext = ref.extents()
    aspect = ext.max(axis=1) / ext.min(axis=1)
    long_u = ext[:, 0] >= ext[:, 1]
    if spec.direction == "auto":
        thin = aspect > spec.aspect_target
        n_long = np.minimum(
            spec.n_max, np.ceil(aspect / spec.aspect_target).astype(np.int64)
        )
        n_long = np.maximum(n_long, 2)
        nu = np.where(thin, np.where(long_u, n_long, 1), 2)
        nv = np.where(thin, np.where(long_u, 1, n_long), 2)
    else:
        nu = np.full(n, 2, dtype=np.int64)
        nv = np.full(n, 2, dtype=np.int64)
    # Respect the accumulated factor cap in each direction.
    room_u = np.maximum(spec.max_factor // factors[:, 0], 1)
    room_v = np.maximum(spec.max_factor // factors[:, 1], 1)
    nu = np.minimum(nu, room_u)
    nv = np.minimum(nv, room_v)
    splittable = nu * nv > 1

    pool = ind[candidates]
    if spec.threshold is not None:
        level = float(spec.threshold)
    elif spec.quantile is not None:
        level = float(np.quantile(pool, spec.quantile)) if pool.size else np.inf
    else:
        k = int(np.ceil(spec.top_fraction * pool.size))
        if pool.size == 0:
            level = np.inf
        elif k >= pool.size:
            level = -np.inf
        else:
            level = float(np.sort(pool)[::-1][k])
    marked = candidates & (ind > level)
    n_at_max = int(np.count_nonzero(marked & ~splittable))
    marked &= splittable

    cells = np.flatnonzero(marked)
    n_capped = 0
    if spec.max_cells is not None and cells.size:
        order = cells[np.argsort(-ind[cells], kind="stable")]
        added = np.cumsum(nu[order] * nv[order] - 1)
        keep = int(np.searchsorted(added, spec.max_cells, side="right"))
        n_capped = cells.size - keep
        cells = np.sort(order[:keep])

    return Marking(
        cells=cells,
        nu=nu[cells],
        nv=nv[cells],
        indicator=indicator,
        level=level,
        n_candidates=int(np.count_nonzero(candidates)),
        n_non_quad=int(np.count_nonzero(~quad)),
        n_nonfinite=int(np.count_nonzero(quad & ~finite)),
        n_at_max_factor=n_at_max,
        n_capped=n_capped,
    )


def apply(ref, fields, spec):
    """Mark by ``spec`` and split.

    Returns
    -------
    (source, marking)
        ``source`` maps every new leaf to its cell before the call (see
        :meth:`~pyaeroflow.mesh.refine.Refinement.split`), or ``None`` if no
        cell was marked; ``marking`` is the :class:`Marking`.
    """
    marking = plan(ref, fields, spec)
    if marking.cells.size == 0:
        return None, marking
    source = ref.split(
        marking.cells,
        marking.nu,
        marking.nv,
        max_hanging=spec.max_hanging,
        column_aspect=spec.column_aspect,
    )
    return source, marking
