"""PyAeroFlow: 2D unstructured finite-volume Navier-Stokes/RANS solver in JAX.

Importing the package switches JAX to 64-bit floating point
(``jax_enable_x64``). This is a global JAX setting and affects all JAX code in
the same Python process.
"""

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)


def _enable_compilation_cache():
    """Keep compiled XLA programs on disk so a new process does not recompile.

    Every ``pyaeroflow run`` or polar process otherwise compiles the same
    kernels again. The cache is keyed by program, JAX version and platform, so
    a changed case or mesh size only compiles what is new. Directory:
    ``$PYAEROFLOW_CACHE_DIR/jax``, else ``~/.cache/pyaeroflow/jax``. An own
    ``JAX_COMPILATION_CACHE_DIR`` is respected; ``PYAEROFLOW_JAX_CACHE=0``
    switches it off. Failures (read-only home) are ignored.
    """
    if os.environ.get("PYAEROFLOW_JAX_CACHE", "1") == "0":
        return
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return
    try:
        root = os.environ.get("PYAEROFLOW_CACHE_DIR")
        base = Path(root) if root else Path.home() / ".cache" / "pyaeroflow"
        cache = base / "jax"
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    except Exception:  # noqa: BLE001 - the cache is an optimisation only
        pass


_enable_compilation_cache()

try:
    __version__ = version("pyaeroflow")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "unknown"
