"""Named layout of the state columns (variable-density energy design, V1).

Every flow family stores its unknowns per cell as columns of one array. The
layout names the blocks of these columns so that code asks for a block by
name instead of a hard-coded column number:

    flow         p, u, v                       (always columns 0-2)
    turbulence   nu_tilde (SA) or k, omega (SST)
    transition   log-intermittency, amplification factor (SA-AFT) or
                 gamma, Re_thetat (SST-LM)
    energy       T                              (incompressible energy equation)
    species      Y_1 .. Y_{N-1}                 (planned: mass transfer)

New blocks are appended after the existing ones, so the turbulence and
transition columns keep their numbers (column 3 is still nu_tilde or k) and
restart files of older runs stay readable. A restart file stores the column
names (:attr:`StateLayout.names`); :func:`map_columns` takes over the
columns that two layouts share by name.
"""

from typing import NamedTuple

FLOW_NAMES = ("p", "u", "v")
TURBULENCE_NAMES = {"none": (), "sa-neg": ("nu_tilde",), "sst": ("k", "omega")}
TRANSITION_NAMES = {
    "none": (),
    "bcm": (),  # algebraic: no transport equation
    "aft": ("log_gamma", "amplification"),
    "lm": ("gamma", "re_theta"),
}


class StateLayout(NamedTuple):
    """Column blocks of a state array (hashable).

    Attributes
    ----------
    blocks : tuple of (str, tuple of str)
        Block name and the names of its columns, in column order.
    """

    blocks: tuple

    @property
    def names(self):
        """All column names in column order."""
        return tuple(n for _, cols in self.blocks for n in cols)

    @property
    def n_var(self):
        """Number of columns."""
        return len(self.names)

    def has(self, block):
        """Return True if the layout has a non-empty block of that name."""
        return any(b == block and cols for b, cols in self.blocks)

    def start(self, block):
        """First column of a block (KeyError if absent or empty)."""
        col = 0
        for b, cols in self.blocks:
            if b == block and cols:
                return col
            col += len(cols)
        raise KeyError(f"state layout has no block '{block}'")

    def slice(self, block):
        """Column slice of a block."""
        start = self.start(block)
        size = len(dict(self.blocks)[block])
        return slice(start, start + size)

    def column(self, name):
        """Column of a variable by name (e.g. "T", "nu_tilde")."""
        return self.names.index(name)


def incompressible_layout(
    turbulence="none", transition="none", energy=False, species=0
):
    """Layout of the incompressible family.

    Parameters
    ----------
    turbulence : {"none", "sa-neg", "sst"}
    transition : {"none", "bcm", "aft", "lm"}
    energy : bool
        Temperature column of the energy equation.
    species : int
        Number of transported mass fractions (planned; 0 today).
    """
    blocks = [
        ("flow", FLOW_NAMES),
        ("turbulence", TURBULENCE_NAMES[turbulence]),
        ("transition", TRANSITION_NAMES[transition]),
        ("energy", ("T",) if energy else ()),
        ("species", tuple(f"Y{i + 1}" for i in range(species))),
    ]
    return StateLayout(tuple(blocks))


def map_columns(stored, target):
    """Pairs (stored column, target column) of the variables both layouts have.

    ``stored`` and ``target`` are sequences of column names.
    """
    index = {name: i for i, name in enumerate(stored)}
    return [(index[name], j) for j, name in enumerate(target) if name in index]
