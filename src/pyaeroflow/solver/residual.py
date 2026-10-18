"""Assembly of the finite-volume residual from face fluxes."""

import jax


def divergence(geom, f_flux, b_flux):
    """Sum face fluxes into cell residuals (net outflow).

    Parameters
    ----------
    geom : FVGeometry
    f_flux : jax array, shape (n_ifaces, n_var)
        Flux density through interior faces, in the direction of the face
        normal (owner to neighbour).
    b_flux : jax array, shape (n_bfaces, n_var)
        Outward flux density through boundary faces.

    Returns
    -------
    jax array, shape (n_cells, n_var)
        R_i = sum over the faces of cell i of flux * length (outward positive).
    """
    n = geom.area.shape[0]
    fl = f_flux * geom.f_length[:, None]
    bl = b_flux * geom.b_length[:, None]
    return (
        jax.ops.segment_sum(fl, geom.f_owner, num_segments=n)
        - jax.ops.segment_sum(fl, geom.f_neigh, num_segments=n)
        + jax.ops.segment_sum(bl, geom.b_owner, num_segments=n)
    )
