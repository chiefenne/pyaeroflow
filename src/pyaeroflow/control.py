"""Control files of a running case: ``.stop``, ``.result_N``, ``.read_toml_N`` (L7).

Both are looked up in the output directory of the case, once per iteration
(steady) or time step (unsteady):

``.stop``
    The run writes a backup and the result files of that iteration or step
    and ends. A run that is started or restarted while the file exists fails
    right away.
``.result_N``
    The run writes the VTU file of iteration or step ``N`` with the fields of
    the case file and goes on. Text in the file names further fields for that
    one file (whitespace or comma separated, ``#`` starts a comment), in
    addition to ``output.fields``. At the start the file only gives a notice.
``.read_toml_N``
    At iteration or step ``N`` the run reads the case file again and applies
    the settings that may change during a run (see :mod:`pyaeroflow.reload`).

The solver never deletes a control file.
"""

import re

STOP = ".stop"
RESULT = ".result_"
READ_TOML = ".read_toml_"
_RESULT = re.compile(r"\.result_(\d+)$")
_READ_TOML = re.compile(r"\.read_toml_(\d+)$")


class StopFileError(ValueError):
    """A run was started while the ``.stop`` file exists."""


def check_no_stop_file(out_dir):
    """Raise :class:`StopFileError` if ``out_dir`` holds a ``.stop`` file."""
    path = out_dir / STOP
    if path.exists():
        raise StopFileError(
            f"the stop file {path} exists: remove it to start the run (rm {path})"
        )


class ControlFiles:
    """The control files of one run in ``out_dir``.

    Parameters
    ----------
    out_dir : Path
        Output directory of the case.
    after : int
        Iteration or step the run starts after (0, or the restart number).
    """

    def __init__(self, out_dir, after=0):
        self.out_dir, self.after = out_dir, after
        self.deferred = set()  # due on a rejected iteration: next good one
        self.done = set()
        self.reload_deferred = set()
        self.reload_done = set()

    def _numbers(self, prefix, pattern):
        found = (pattern.match(p.name) for p in self.out_dir.glob(prefix + "*"))
        return sorted(int(m.group(1)) for m in found if m)

    def result_numbers(self):
        """Numbers N of the ``.result_N`` files that exist now, ascending."""
        return self._numbers(RESULT, _RESULT)

    def reload_numbers(self):
        """Numbers N of the ``.read_toml_N`` files that exist now, ascending."""
        return self._numbers(READ_TOML, _READ_TOML)

    def result_fields(self, number):
        """Field names written in ``.result_<number>`` (empty = none)."""
        try:
            text = (self.out_dir / f"{RESULT}{number}").read_text()
        except OSError:
            return []
        words = []
        for line in text.splitlines():
            words += line.split("#", 1)[0].replace(",", " ").split()
        return words

    def notices(self, unit="iteration"):
        """Notices for the ``.result_N`` files found at the start of the run."""
        lines = []
        for prefix, numbers, what, skipped in (
            (RESULT, self.result_numbers(), "the fields are written", "not written"),
            (READ_TOML, self.reload_numbers(), "the case file is read", "not read"),
        ):
            for n in numbers:
                if n > self.after:
                    lines.append(
                        f"notice: control file {prefix}{n} found, {what} at {unit} {n}"
                    )
                else:
                    lines.append(
                        f"notice: control file {prefix}{n} found, but the run starts "
                        f"after {unit} {self.after}: {skipped}"
                    )
        return lines

    def stop_requested(self):
        """Return True if the ``.stop`` file exists."""
        return (self.out_dir / STOP).exists()

    def due_results(self, number, good=True):
        """Numbers N whose VTU file is due at iteration or step ``number``.

        A file created after its number has passed is ignored. One that falls
        on a rejected iteration or a time step at the inner limit (``good``
        False) waits for the next good one.
        """
        present = set(self.result_numbers())
        self.deferred &= present
        if number in present and number > self.after:
            self.deferred.add(number)
        if not good:
            return []
        due = sorted(n for n in self.deferred if n not in self.done)
        self.done.update(due)
        self.deferred.clear()
        return due

    def due_reloads(self, number, good=True):
        """Numbers N of the ``.read_toml_N`` files due at ``number`` (as results)."""
        present = set(self.reload_numbers())
        self.reload_deferred &= present
        if number in present and number > self.after:
            self.reload_deferred.add(number)
        if not good:
            return []
        due = sorted(n for n in self.reload_deferred if n not in self.reload_done)
        self.reload_done.update(due)
        self.reload_deferred.clear()
        return due
