"""PyAeroFlow: 2D unstructured finite-volume Navier-Stokes/RANS solver in JAX.

Importing the package switches JAX to 64-bit floating point
(``jax_enable_x64``). This is a global JAX setting and affects all JAX code in
the same Python process.
"""

from importlib.metadata import PackageNotFoundError, version

import jax

jax.config.update("jax_enable_x64", True)

try:
    __version__ = version("pyaeroflow")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "unknown"
