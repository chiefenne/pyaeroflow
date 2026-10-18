"""Steady scalar convection-diffusion: the stage-1 model problem.

    div(a u - nu grad u) = 0

with constant velocity a and diffusivity nu, and Dirichlet values on all
boundary faces. It exercises the operators and the implicit machinery used by
the flow solver:

- convective flux: upwind (the scalar form of flux-difference splitting, D21)
  with first-order or MUSCL face states and an optional limiter (D23);
- diffusive flux: least-squares gradients (D22) and the face-gradient scheme
  of D24 (alpha-damping or edge correction);
- compact first-order face fluxes for the approximate Jacobian (D25).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyaeroflow.numerics import (
    boundary_face_gradient,
    interior_face_gradient,
    lsq_gradient,
    reconstruct,
    venkatakrishnan_wang,
)
from pyaeroflow.solver import divergence


class ConvectionDiffusion(NamedTuple):
    """Model parameters (hashable, used as static jit argument).

    Attributes
    ----------
    velocity : tuple of float
        Convection velocity (a_x, a_y).
    diffusivity : float
        Diffusion coefficient nu.
    spatial_order : int
        1: piecewise-constant convective states; 2: MUSCL reconstruction.
    limiter : {"none", "venkatakrishnan-wang"}
    limiter_threshold : float
        Wang's epsilon'.
    viscous_gradient : {"alpha-damping", "edge-correction"}
    alpha : float
        Damping coefficient of the alpha-damping scheme.
    """

    velocity: tuple = (1.0, 0.5)
    diffusivity: float = 0.25
    spatial_order: int = 2
    limiter: str = "none"
    limiter_threshold: float = 0.05
    viscous_gradient: str = "alpha-damping"
    alpha: float = 4.0 / 3.0


@partial(jax.jit, static_argnums=0)
def residual(model, geom, u, u_b):
    """Second-order residual R(u), shape (n_cells, 1).

    Parameters
    ----------
    model : ConvectionDiffusion
    geom : FVGeometry
    u : jax array, shape (n_cells, 1)
    u_b : jax array, shape (n_bfaces, 1)
        Dirichlet values at the boundary-face centres.
    """
    a = jnp.asarray(model.velocity)
    grad = lsq_gradient(geom, u, u_b)
    if model.spatial_order == 2:
        if model.limiter == "venkatakrishnan-wang":
            psi = venkatakrishnan_wang(geom, u, u_b, grad, model.limiter_threshold)
        elif model.limiter == "none":
            psi = jnp.ones_like(u)
        else:
            raise ValueError(f"unknown limiter '{model.limiter}'")
        u_left, u_right, _ = reconstruct(geom, u, grad, psi)
    elif model.spatial_order == 1:
        u_left, u_right = u[geom.f_owner], u[geom.f_neigh]
    else:
        raise ValueError("spatial_order must be 1 or 2")

    an = (geom.f_normal @ a)[:, None]
    convective = an * jnp.where(an >= 0.0, u_left, u_right)
    g_f = interior_face_gradient(geom, u, grad, model.viscous_gradient, model.alpha)
    diffusive = -model.diffusivity * jnp.einsum("fvk,fk->fv", g_f, geom.f_normal)

    g_b = boundary_face_gradient(geom, u, u_b, grad, model.viscous_gradient)
    b_flux = (geom.b_normal @ a)[:, None] * u_b - model.diffusivity * jnp.einsum(
        "fvk,fk->fv", g_b, geom.b_normal
    )
    return divergence(geom, convective + diffusive, b_flux)


def interior_flux(model, params, u_o, u_n, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian).

    Upwind convection and the two-point part of the face-gradient scheme.
    ``params`` is unused (the model has no per-iteration parameters).
    """
    an = jnp.dot(jnp.asarray(model.velocity), normal)
    dn = jnp.dot(delta, normal)
    if model.viscous_gradient == "alpha-damping":
        coeff = model.alpha / dn
    else:  # edge correction: (u_n - u_o) / |d| * (e . n)
        coeff = dn / jnp.dot(delta, delta)
    return an * jnp.where(an >= 0.0, u_o, u_n) - model.diffusivity * coeff * (u_n - u_o)


def boundary_flux(model, params, u_o, u_b, normal, delta):
    """Compact first-order flux through a Dirichlet boundary face."""
    an = jnp.dot(jnp.asarray(model.velocity), normal)
    dn = jnp.dot(delta, normal)
    return an * u_b - model.diffusivity * (u_b - u_o) / dn


def spectral_radius(model, geom):
    """Sum over faces of (|a.n| + nu / (d.n)) * length, per cell."""
    a = np.asarray(model.velocity)
    n = geom.area.shape[0]
    f_normal, b_normal = np.asarray(geom.f_normal), np.asarray(geom.b_normal)
    lam_f = (
        np.abs(f_normal @ a)
        + model.diffusivity / np.einsum("fk,fk->f", np.asarray(geom.f_delta), f_normal)
    ) * np.asarray(geom.f_length)
    lam_b = (
        np.abs(b_normal @ a)
        + model.diffusivity / np.einsum("fk,fk->f", np.asarray(geom.b_delta), b_normal)
    ) * np.asarray(geom.b_length)
    return (
        np.bincount(np.asarray(geom.f_owner), lam_f, n)
        + np.bincount(np.asarray(geom.f_neigh), lam_f, n)
        + np.bincount(np.asarray(geom.b_owner), lam_b, n)
    )


def exact_solution(model, points, reference=(1.0, 1.0)):
    """Exact solution u = exp(a . (x - x_ref) / nu), shape (n_points, 1).

    It satisfies a . grad u = nu laplace u, i.e. div(a u - nu grad u) = 0 for
    constant a; the reference point only scales the solution.
    """
    a = np.asarray(model.velocity)
    shift = (np.asarray(points) - np.asarray(reference)) @ a
    return np.exp(shift / model.diffusivity)[:, None]
