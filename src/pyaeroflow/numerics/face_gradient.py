"""Face gradients for viscous fluxes (requirement D24).

Both schemes start from the average of the two cell gradients and add a term
on the solution jump between the cell centroids j (owner) and k (neighbour):

    jump = (u_k - u_j) - g_avg . (x_k - x_j)

- ``"alpha-damping"`` (Nishikawa 2010; damping term at the midpoint between
  the cell centroids, Nishikawa 2026):
  grad_f = g_avg + alpha / ((x_k - x_j) . n) * jump * n
- ``"edge-correction"`` (fallback):
  grad_f = g_avg + jump / |x_k - x_j| * e,  e = (x_k - x_j) / |x_k - x_j|

At boundary faces the face centre and the boundary value replace the
neighbour, the cell gradient replaces the average, and alpha is 1, so the
jump term supplies the one-sided normal derivative.

Both schemes are exact for linear fields.
"""

from functools import partial

import jax
import jax.numpy as jnp

SCHEMES = ("alpha-damping", "edge-correction")


def _corrected(g, du, d, normal, alpha, scheme):
    """Average gradient plus jump correction, for arrays of faces."""
    jump = du - jnp.einsum("fvk,fk->fv", g, d)
    if scheme == "alpha-damping":
        dn = jnp.einsum("fk,fk->f", d, normal)
        return g + (alpha * jump / dn[:, None])[:, :, None] * normal[:, None, :]
    if scheme == "edge-correction":
        length = jnp.linalg.norm(d, axis=1)
        e = d / length[:, None]
        return g + (jump / length[:, None])[:, :, None] * e[:, None, :]
    raise ValueError(f"viscous gradient scheme must be one of {SCHEMES}")


@partial(jax.jit, static_argnames="scheme")
def interior_face_gradient(geom, u, grad, scheme="alpha-damping", alpha=4.0 / 3.0):
    """Face gradients at interior faces, shape (n_ifaces, n_var, 2)."""
    o, n = geom.f_owner, geom.f_neigh
    g = 0.5 * (grad[o] + grad[n])
    return _corrected(g, u[n] - u[o], geom.f_delta, geom.f_normal, alpha, scheme)


@partial(jax.jit, static_argnames="scheme")
def boundary_face_gradient(geom, u, u_b, grad, scheme="alpha-damping"):
    """Face gradients at boundary faces, shape (n_bfaces, n_var, 2)."""
    b = geom.b_owner
    return _corrected(grad[b], u_b - u[b], geom.b_delta, geom.b_normal, 1.0, scheme)
