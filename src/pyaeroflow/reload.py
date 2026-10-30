"""Reading the case file again during a run: ``.read_toml_N`` (L7).

At iteration or step ``N`` the run loads the case file anew and compares it
with the settings it uses. Differences in :data:`LIVE_STEADY` (steady runs) or
:data:`LIVE_UNSTEADY` (unsteady runs) are applied; any other difference
(physics, mesh, boundary conditions, numerics) is reported and not applied,
because the run's state would no longer belong to the case. Every reload is
written to the log and a copy of the settings in force afterwards goes to
``<name>.read_toml_<N>.toml``, so a run can be repeated from its log. A case
file that no longer validates leaves the settings as they are.
"""

OUTPUT_KEYS = (
    "output.fields",
    "output.fields_interval",
    "output.fields_at",
    "output.fields_final",
    "output.restart_interval",
    "output.restart_backup_interval",
    "output.restart_backup_at",
)
LIVE_UNSTEADY = OUTPUT_KEYS
LIVE_STEADY = OUTPUT_KEYS + (
    "solver.max_iterations",
    "solver.residual_drop",
    "solver.convergence",
)


class Reload:
    """Outcome of one reload: what was applied, what was not, or the error."""

    def __init__(self, applied=(), ignored=(), error=None):
        self.applied, self.ignored, self.error = list(applied), list(ignored), error

    def changed(self, prefix):
        """Return True if an applied key is ``prefix`` or lies below it."""
        return any(
            k == prefix or k.startswith(prefix + ".") for k, _, _ in self.applied
        )

    def lines(self, number, unit):
        """Log lines of the reload."""
        head = f"control file .read_toml_{number}: case file read at {unit} {number}"
        if self.error:
            return [head, f"  not applied, the case file is invalid: {self.error}"]
        if not self.applied and not self.ignored:
            return [head + ", no changes"]
        lines = [head]
        lines += [
            f"  changed {k}: {_show(a)} -> {_show(b)}" for k, a, b in self.applied
        ]
        lines += [
            f"  not applied (cannot change during a run) {k}: {_show(a)} -> {_show(b)}"
            for k, a, b in self.ignored
        ]
        return lines


def _show(value):
    return "(unset)" if value is None else str(value)


def flatten(data, prefix=""):
    """``{dotted.key: value}`` of the leaves of a nested dict."""
    out = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten(value, path + "."))
        else:
            out[path] = value
    return out


ALIASES = {"output.interval": "output.fields_interval"}  # mirrored by the resolver


def _is_live(key, live):
    return any(key == k or key.startswith(k + ".") for k in live)


def _set(data, key, value):
    *parents, last = key.split(".")
    for part in parents:
        data = data[part]
    data[last] = value


def reload_case(config, live):
    """Read the case file again and apply the live changes to ``config``.

    ``config.reload`` returns the freshly resolved case (see
    :func:`pyaeroflow.config.load_case`). The values in ``config.data`` change
    in place for the keys in ``live``.
    """
    if config.reload is None:
        return Reload(error="this run has no case file to read")
    try:
        new = config.reload()
    except Exception as exc:  # a typo in a running case must not end the run
        return Reload(error=str(exc).strip().replace("\n", "; "))
    old, fresh = flatten(config.data), flatten(new.data)
    applied, ignored = [], []
    for key in sorted(set(old) | set(fresh)):
        if old.get(key) == fresh.get(key):
            continue
        if key in ALIASES:  # follows its target, not a change of its own
            _set(config.data, key, fresh[key])
            continue
        entry = (key, old.get(key), fresh.get(key))
        if _is_live(key, live) and key in old and key in fresh:
            _set(config.data, key, fresh[key])
            applied.append(entry)
        else:
            ignored.append(entry)
    return Reload(applied, ignored)
