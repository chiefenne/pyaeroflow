#!/usr/bin/env python3
"""Plot a solver log: kept for existing commands, now ``pyaeroflow plot log``.

Usage:  python plot_log.py <logfile> [-t targets.toml] [-o output_prefix] [--no-notes]

Same as ``pyaeroflow plot log <logfile> ...``; see ``pyaeroflow plot --help``
and pyaeroflow.plotting.log for the plots and the targets file.
"""

import sys

from pyaeroflow.cli import main

if __name__ == "__main__":
    sys.exit(main(["plot", "log", *sys.argv[1:]]))
