"""Control files of a running case: ``.stop`` and ``.result_N`` (L7).

Both are looked up in the output directory of the case, once per iteration
(steady) or time step (unsteady):

``.stop``
    The run writes a backup and the result files of that iteration or step
    and ends. A run that is started or restarted while the file exists fails
    right away.
``.result_N``
    The run writes the VTU file of iteration or step ``N`` with the fields of
    the case file and goes on. At the start the file only gives a notice. The
    solver never deletes either file.
"""

import re

STOP = ".stop"
RESULT = ".result_"
_RESULT = re.compile(r"\.result_(\d+)$")


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

    def result_numbers(self):
        """Numbers N of the ``.result_N`` files that exist now, ascending."""
        found = (_RESULT.match(p.name) for p in self.out_dir.glob(RESULT + "*"))
        return sorted(int(m.group(1)) for m in found if m)

    def notices(self, unit="iteration"):
        """Notices for the ``.result_N`` files found at the start of the run."""
        lines = []
        for n in self.result_numbers():
            if n > self.after:
                lines.append(
                    f"notice: control file {RESULT}{n} found, the fields are "
                    f"written at {unit} {n}"
                )
            else:
                lines.append(
                    f"notice: control file {RESULT}{n} found, but the run starts "
                    f"after {unit} {self.after}: not written"
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
