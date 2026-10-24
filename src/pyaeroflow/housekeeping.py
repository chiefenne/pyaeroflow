"""Names of the saved-state files, their inspection and cleaning (D56, D60).

A run writes four kinds of state files next to its log and history:

* backups ``<name>.backup_<number, 6 digits>.npz``: never replaced, written by
  ``output.restart_backup_interval`` and ``restart_backup_at`` and once more at
  the end of the run, so the highest number is where the run stopped;
* rolling restart files ``<name>.restart_a.npz`` and ``<name>.restart_b.npz``
  (``output.restart_interval``), each write replacing the older one;
* fields ``<name>.vtu``, ``<name>_<number>.vtu`` and ``<name>.pvd``;
* history, wall CSVs, the resolved configuration and the log.

Backups and rolling files have the same content; any of them can be the
``initial.restart`` of a continuation. Runs of earlier versions wrote
``<name>.restart.npz`` (final) and ``<name>.restart_<number>.npz`` (backups);
those are still read, and cleaned as backups.
"""

import re
from pathlib import Path

import numpy as np

BACKUP_DIGITS = 6


def backup_name(name, number):
    """File name of the backup of ``name`` at iteration or step ``number``."""
    return f"{name}.backup_{int(number):0{BACKUP_DIGITS}d}.npz"


def backup_number(path):
    """Return the iteration or step in a backup file name, None for other files."""
    m = re.search(r"\.(?:backup|restart)_(\d{6})\.npz$", Path(path).name)
    return int(m[1]) if m else None


def latest_backup(directory, name):
    """Return the backup of ``name`` with the highest number, or None.

    Falls back to the ``<name>.restart.npz`` of a run of an earlier version.
    """
    directory = Path(directory)
    found = [
        (n, p)
        for p in directory.glob(f"{name}.backup_*.npz")
        if (n := backup_number(p)) is not None
    ]
    if found:
        return max(found)[1]
    legacy = directory / f"{name}.restart.npz"
    return legacy if legacy.exists() else None


def restart_info(path):
    """Describe a restart or backup file: where the run was and what it stored.

    Returns
    -------
    dict
        ``file``, ``kind`` ("steady" or "unsteady", None for a file without
        solver state), ``iteration`` or ``step``, ``time`` (s, unsteady),
        ``n_cells``, ``n_variables``, ``fingerprint`` (of the mesh),
        ``version`` (of PyAeroFlow that wrote it) and the scalars of the stored
        solver state (``cfl``, ``eta``, ``elapsed`` ...).
    """
    path = Path(path)
    info = {"file": str(path)}
    with np.load(path) as data:
        files = data.files
        info["n_cells"], info["n_variables"] = (
            int(data["n_cells"]) if "n_cells" in files else None,
            int(data["w"].shape[1]),
        )
        info["fingerprint"] = (
            str(data["fingerprint"]) if "fingerprint" in files else None
        )
        info["version"] = str(data["version"]) if "version" in files else None
        kind = str(data["state_kind"]) if "state_kind" in files else None
        info["kind"] = kind
        step = int(data["step"]) if "step" in files else None
        info["step" if kind == "unsteady" else "iteration"] = step
        if "time" in files:
            info["time"] = float(data["time"])
        for key in files:
            if not key.startswith("state_") or key == "state_kind":
                continue
            value = np.array(data[key])
            if value.size == 1 and value.dtype.kind in "fiub":
                info[key[6:]] = value.item()
    return info


# Categories of ``clean``: name -> regular expression of the file names, with
# ``{p}`` the case name or ``<case name>_alpha_<angle>`` of a polar run.
CATEGORIES = {
    "restarts": (r"{p}\.restart_[ab]\.npz",),
    "backups": (
        r"{p}\.backup_\d{{6}}\.npz",
        r"{p}\.restart_\d{{6}}\.npz",
        r"{p}\.restart\.npz",
    ),
    "fields": (r"{p}\.vtu", r"{p}_\d{{6}}\.vtu", r"{p}\.pvd"),
    "history": (r"{p}\.history\.csv", r"{p}\.history\.cut_\d+\.csv"),
    "wall": (r"{p}\.wall_.+\.csv",),
    "log": (r"{p}\.log",),
}
# Only with --all: the rest of what a run writes.
ALL_EXTRA = (r"{p}\.resolved\.toml", r"{p}\.polar\.csv")


def case_files(out_dir, name):
    """Files of a case in ``out_dir`` (and its ``alpha_*`` folders), by category.

    Returns
    -------
    dict of str to list of Path
        Keys: the categories above and ``"other"`` (resolved configuration,
        polar CSV).
    """
    out_dir = Path(out_dir)
    prefix = rf"{re.escape(name)}(?:_alpha_[-+0-9.eE]+)?"
    folders = [out_dir] + sorted(p for p in out_dir.glob("alpha_*") if p.is_dir())
    found = {key: [] for key in (*CATEGORIES, "other")}
    for folder in folders:
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            for key, patterns in {**CATEGORIES, "other": ALL_EXTRA}.items():
                if any(
                    re.fullmatch(pat.format(p=prefix), path.name) for pat in patterns
                ):
                    found[key].append(path)
                    break
    return found


def clean_case(out_dir, name, selected, delete=False):
    """List, and with ``delete`` remove, the files of the selected categories.

    Parameters
    ----------
    out_dir : Path
        Output directory of the case.
    name : str
        Case name (``case.name``).
    selected : iterable of str
        Categories to remove: any of ``restarts``, ``backup``, ``fields``,
        ``history``, ``wall``, ``log`` and ``other``.
    delete : bool
        False: only report.

    Returns
    -------
    inventory : dict of str to list of Path
        Every category with its files.
    removed : list of Path
        The files deleted (empty unless ``delete``).
    """
    found = case_files(out_dir, name)
    removed = []
    if delete:
        for key in selected:
            for path in found.get(key, ()):
                path.unlink()
                removed.append(path)
        for folder in sorted(Path(out_dir).glob("alpha_*")):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
    return found, removed
