"""Compiled kernels of the viscous panel method, kept on disk between runs.

``jax.jit`` compiles a function in two steps: Python traces it into a program
(StableHLO), XLA compiles the program. The persistent compilation cache of
pyaeroflow (``pyaeroflow/__init__.py``) keeps the second step. For the
boundary-layer kernels the first step costs as much: the closures are traced
through ``jacfwd`` and ``vmap`` in every new process. :func:`cached_jit`
keeps the traced program too (``jax.export``), in ``<cache>/ibl``, keyed by
the function name, the argument shapes, the JAX version and the source of
this package. Any failure falls back to a plain ``jax.jit``.
"""

import hashlib
import os
from pathlib import Path

import jax
import numpy as np

__all__ = ["cached_jit", "cache_directory"]


def cache_directory():
    """Directory of the traced programs, or None when the cache is off."""
    if os.environ.get("PYAEROFLOW_JAX_CACHE", "1") == "0":
        return None
    root = os.environ.get("PYAEROFLOW_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "pyaeroflow"
    return base / "ibl"


_FINGERPRINT = None


def _fingerprint():
    """Hash of the JAX version and of the sources the kernels are traced from."""
    global _FINGERPRINT
    if _FINGERPRINT is None:
        h = hashlib.sha256(jax.__version__.encode())
        here = Path(__file__).parent
        files = sorted(here.glob("*.py")) + [here.parent / "numerics" / "panel.py"]
        for f in files:
            h.update(f.read_bytes())
        _FINGERPRINT = h.hexdigest()[:16]
    return _FINGERPRINT


def _small_compiles():
    """Let the XLA cache also keep the many small programs of the panel set-up."""
    try:
        if jax.config.jax_compilation_cache_dir:
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception:  # noqa: BLE001 - the cache is an optimisation only
        pass


def _load(name, signature, jitted, args):
    directory = cache_directory()
    if directory is None:
        return jitted
    from jax import export

    key = hashlib.sha256(f"{name}|{signature}|{_fingerprint()}".encode()).hexdigest()[
        :24
    ]
    path = directory / f"{name}-{key}.bin"
    try:
        if path.exists():
            exported = export.deserialize(bytearray(path.read_bytes()))
        else:
            exported = export.export(jitted)(*args)
            directory.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(bytes(exported.serialize()))
            os.replace(tmp, path)
        fn = jax.jit(exported.call)
        fn(*args)  # a program that does not fit the arguments fails here
        return fn
    except Exception:  # noqa: BLE001 - fall back to tracing in this process
        return jitted


def cached_jit(name, fn):
    """``jax.jit(fn)`` whose traced program is stored on disk.

    The arguments are converted to arrays (a Python float becomes a float64
    scalar), so one stored program serves every call with the same shapes.
    """
    jitted = jax.jit(fn)
    table = {}

    def call(*args):
        leaves, tree = jax.tree_util.tree_flatten(args)
        leaves = [x if hasattr(x, "dtype") else np.asarray(x) for x in leaves]
        signature = (tree, tuple((x.shape, str(x.dtype)) for x in leaves))
        args = jax.tree_util.tree_unflatten(tree, leaves)
        f = table.get(signature)
        if f is None:
            _small_compiles()
            f = table[signature] = _load(name, str(signature), jitted, args)
        return f(*args)

    return call
