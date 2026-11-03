"""Set up and solve a flow problem (Python API, D41).

:class:`FlowProblem` binds a mesh, a model and boundary conditions and
provides the residual, Jacobian and pseudo-time term for the solvers in
:mod:`pyaeroflow.solver`. The model selects the flow family (L1):
:class:`~pyaeroflow.physics.incompressible.IncompressibleModel` (SU2 INC,
with the turbulence and transition models) or
:class:`~pyaeroflow.physics.compressible.CompressibleModel` (density-based,
Euler, laminar Navier-Stokes and SA-neg RANS). A flow family is a physics
module with the same interface:
``build_boundary_data``, ``lsq_include``, ``scheme_parameters`` (the
per-evaluation constants of the scheme that the residual, Jacobian and
pseudo-time term take frozen; beta^2 of the artificial compressibility for
INC, none for the compressible solver), ``residual``, ``residual_jvp``,
``residual_linearize``, ``limiter``, ``boundary_state``, ``interior_flux``,
``boundary_flux``, ``time_term_blocks``, ``explicit_update`` and the
constants ``FLOW``, ``POSITIVE_COLUMNS`` and ``RESIDUAL_NAMES``.
:class:`IncompressibleProblem` is the older name of :class:`FlowProblem`.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pyaeroflow.numerics import build_geometry, lsq_gradient
from pyaeroflow.numerics.wall_distance import mesh_wall_distance, mesh_wall_normal
from pyaeroflow.physics import compressible as comp
from pyaeroflow.physics import forces as fz
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import lm, rans, rans_compressible, rans_sst, sst
from pyaeroflow.solver import (
    JacobianPattern,
    PseudoTimeSettings,
    block_diagonal,
    first_order_jacobian,
    solve_explicit,
    solve_steady,
    solve_unsteady,
    structural_masks,
)

PRESSURE_FIXING = {"pressure-outlet", "farfield", "total-pressure-inlet"}
INFLOW_TYPES = {"velocity-inlet", "total-pressure-inlet", "farfield"}
TURBULENCE_MODELS = ("sa-neg", "sst")
# Default SST inflow of the Python API: Tu = 0.1 %, nu_t / nu = 1.
SST_DEFAULT_INTENSITY = 0.001
SST_DEFAULT_VISCOSITY_RATIO = 1.0
# SU2's MAX_UPDATE_SST: an update may lower k or omega by at most this
# fraction of the current value (the cell's turbulence update is scaled).
SST_MAX_UPDATE = 0.99
# SU2's limits of k and omega (CTurbSSTSolver lowerlimit/upperlimit, applied
# with AddClippedSolution after every update), non-dimensional here.
SST_K_LIMITS = (1.0e-10, 1.0e10)
SST_OMEGA_LIMITS = (1.0e-4, 1.0e15)
# Compressible flow: an update may lower p or T by at most this fraction of
# the current value (the cell's update is scaled), so that they stay positive.
COMPRESSIBLE_MAX_DROP = 0.8


def sst_default_values(nu, speed):
    """Default SST (k, omega) for the inflow speed (Tu 0.1 %, nu_t/nu = 1)."""
    k = 1.5 * (SST_DEFAULT_INTENSITY * max(speed, 1e-12)) ** 2
    return sst.freestream(SST_DEFAULT_VISCOSITY_RATIO, k, nu)


def flow_module(model):
    """Physics module of the model's flow family (incompressible or compressible)."""
    return comp if isinstance(model, comp.CompressibleModel) else inc


def _inflow_speed(bc):
    vel = bc.get("velocity")
    if vel is None or callable(vel):
        return 1.0
    return float(np.hypot(*vel))


class FarfieldVortex:
    """State of the point-vortex farfield correction (P10).

    Attributes
    ----------
    faces : ndarray
        Boundary-face indices of the farfield faces.
    base : ndarray, shape (n, 3)
        Freestream (p, u, v) of these faces.
    offset : ndarray, shape (n, 2)
        Face centre minus vortex centre.
    boundaries : list of str
        Boundaries whose lift sets the circulation.
    relaxation : float
        Under-relaxation of the circulation update per iteration.
    gamma : float
        Current circulation.
    """

    def __init__(self, faces, base, offset, boundaries, relaxation):
        self.faces = faces
        self.base = base
        self.offset = offset
        self.boundaries = boundaries
        self.relaxation = relaxation
        self.gamma = 0.0


# The diagnostics count cells within this wall distance (non-dimensional,
# chords): far from the wall Re_v = d^2 |omega| / nu is large and gamma flips
# in the freestream, which would swamp the boundary-layer statistics.
GAMMA_DIAGNOSTICS_DISTANCE = 0.05

# Per-iteration transition diagnostics of :meth:`FlowProblem.solve`:
# cells with gamma > 0.01 (within GAMMA_DIAGNOSTICS_DISTANCE of a wall),
# area-weighted sum of gamma, cells whose model
# intermittency changed by more than 0.05 since the last accepted iteration,
# the largest change and its cell centre (x, y), the largest |R_nu| and its
# cell centre. Coordinates are non-dimensional (divided by the length scale).
GAMMA_DIAGNOSTICS = (
    "gamma_cells",
    "gamma_area",
    "gamma_jumped",
    "gamma_dmax",
    "gamma_dmax_x",
    "gamma_dmax_y",
    "rnu_max",
    "rnu_max_x",
    "rnu_max_y",
)


# The first six of them (no residual) are the per-time-step diagnostics of
# unsteady runs, relative to the previous step.
GAMMA_STEP_DIAGNOSTICS = GAMMA_DIAGNOSTICS[:6]


@jax.jit
def _apply_linear(lin, dw):
    """Evaluate a ``jax.linearize`` map (compiled once per structure)."""
    return lin(dw)


class FlowProblem:
    """A flow case on a mesh.

    Parameters
    ----------
    mesh : Mesh
    model : IncompressibleModel or CompressibleModel
        The model's type selects the flow family (:func:`flow_module`).
        If ``model.pin`` is None and no boundary fixes the pressure level
        (pressure outlet or farfield), the pressure of the cell nearest
        ``pressure_point`` is pinned to ``pin_pressure`` (D39 R2).
    boundary_conditions : dict of str to dict
        See :func:`pyaeroflow.physics.incompressible.boundary_data` or
        :func:`pyaeroflow.physics.compressible.boundary_data`. Periodic
        pairs are given as ``{"type": "periodic", "partner": name,
        "translation": (dx, dy)}`` on one of the two boundaries.
    weight_exponent : float
        Least-squares weight exponent (D22).
    pressure_point : tuple of float, optional
        Location of the pinned cell; default the domain centroid.
    laminar_regions : list of dict, optional
        Prescribed laminar regions (D15) for a turbulence model: SA
        production is switched off in cells whose nearest wall face belongs
        to ``boundary`` (optionally restricted to ``side`` "upper"/"lower",
        the side on which the fluid lies above/below the wall) and whose
        centroid lies within the optional ``x_min``, ``x_max``, ``y_min``,
        ``y_max`` bounds.
    pin_pressure : float
        Pinned pressure value.
    """

    def __init__(
        self,
        mesh,
        model,
        boundary_conditions,
        weight_exponent=1.0,
        pressure_point=None,
        pin_pressure=0.0,
        laminar_regions=None,
    ):
        periodic, regular = [], {}
        for name, bc in boundary_conditions.items():
            if bc["type"] == "periodic":
                periodic.append((name, bc["partner"], np.asarray(bc["translation"])))
            else:
                regular[name] = bc
        for name_a, name_b, _ in periodic:
            if name_b in regular:
                raise ValueError(
                    f"periodic partner '{name_b}' of '{name_a}' must not have its "
                    "own boundary condition"
                )
        missing = set(mesh.boundary_faces) - set(regular) - {b for _, b, _ in periodic}
        missing -= {a for a, _, _ in periodic}
        unknown = set(regular) - set(mesh.boundary_faces)
        if missing or unknown:
            raise ValueError(
                f"boundary conditions missing for {sorted(missing)}, "
                f"unknown boundaries {sorted(unknown)}"
            )
        self.mesh = mesh
        self.flow = flow_module(model)
        self.compressible = self.flow is comp
        self.turbulent = model.turbulence != "none"
        if self.compressible and self.turbulent:
            if model.turbulence == "sst":
                raise ValueError("compressible flow: SST is planned (C4)")
            if model.transition != "none":
                raise ValueError("compressible flow: transition models are planned")
        if self.turbulent:
            if model.turbulence not in TURBULENCE_MODELS:
                raise ValueError(f"unknown turbulence model '{model.turbulence}'")
            if model.viscosity <= 0.0:
                raise ValueError("a turbulence model needs a positive viscosity")
            if model.turbulence == "sst" and model.transition not in ("none", "lm"):
                raise ValueError(f"transition model '{model.transition}' needs SA")
            if model.turbulence != "sst" and model.transition == "lm":
                raise ValueError("the Langtry-Menter transition model needs SST")
            nu = model.viscosity / model.density  # compressible: of the reference
            # Default inflow values: nu_tilde = 3 nu (NASA TMR, D33); SST
            # k and omega from Tu = 0.1 %, nu_t / nu = 1.
            defaults = {}
            for name, bc in regular.items():
                if bc["type"] not in INFLOW_TYPES:
                    continue
                if model.turbulence == "sst":
                    k, om = sst_default_values(nu, _inflow_speed(bc))
                    defaults[name] = {"k": k, "omega": om}
                else:
                    defaults[name] = {"nu_tilde": 3.0 * nu}
            regular = {
                name: {**defaults.get(name, {}), **bc} for name, bc in regular.items()
            }
        self.disc = build_geometry(
            mesh, weight_exponent, periodic, self.flow.lsq_include(regular, model.n_var)
        )
        self.geom = self.disc.geom
        if self.turbulent:
            walls = [
                mesh.boundary_faces[n]
                for n, bc in regular.items()
                if bc["type"] in ("wall", "transpiration-wall")
            ]
            faces = np.concatenate(walls) if walls else np.zeros(0, dtype=int)
            distance, nearest = mesh_wall_distance(mesh, faces)
            self.nearest_wall_face = nearest
            self.geom = self.geom._replace(
                wall_distance=jnp.asarray(distance),
                production_switch=jnp.asarray(
                    self._production_switch(mesh, regular, laminar_regions or [])
                ),
            )
            if model.transition == "aft":
                self.geom = self.geom._replace(
                    wall_normal=jnp.asarray(mesh_wall_normal(mesh, nearest))
                )
        self.bc = self.flow.build_boundary_data(self.disc, regular, model)
        if model.turbulence == "sst":
            self._set_wall_omega(model)
            if model.transition == "lm":
                self._set_transition_inflow(regular)
        self.conditions = regular
        self.periodic = periodic
        self.check_mass_balance()

        fixes_pressure = any(bc["type"] in PRESSURE_FIXING for bc in regular.values())
        self.pinned_cell = None
        if model.pin is None and not fixes_pressure:
            point = (
                np.average(mesh.cell_centroid, axis=0, weights=mesh.cell_area)
                if pressure_point is None
                else np.asarray(pressure_point, dtype=float)
            )
            cell = int(np.argmin(np.sum((mesh.cell_centroid - point) ** 2, axis=1)))
            model = model._replace(pin=(cell, float(pin_pressure)))
            self.pinned_cell = cell
        self.model = model
        if self.compressible:
            self.physics = rans_compressible if self.turbulent else comp
        else:
            self.physics = {"sa-neg": rans, "sst": rans_sst}.get(
                model.turbulence, self.flow
            )
        # Column of the first turbulence variable (nu_tilde or k).
        self.turbulence_column = getattr(self.physics, "NT", rans.NT)
        self._pattern = None  # Jacobian sparsity, built at the first assembly
        self._fi = partial(self.physics.interior_flux, model)
        self._fb = partial(self.physics.boundary_flux, model)

    def _set_wall_omega(self, model):
        """SST wall values: k = 0, omega = 60 nu / (beta_1 d1^2) per wall face.

        d1 is the height of the wall cell, twice the wall-normal distance of
        its centroid: on the same mesh points SU2 (vertex-based) uses the
        distance from the wall node to its nearest neighbour, the first mesh
        spacing. The skin friction depends on this choice (about 2 % lower
        with d1 = the centroid distance, on the flat plate at every grid
        level): the 1/y^2 profile of omega is not resolved by the wall cell.
        """
        nu = model.viscosity / model.density
        delta, normal = np.asarray(self.geom.b_delta), np.asarray(self.geom.b_normal)
        d1 = 2.0 * np.abs(np.einsum("fk,fk->f", delta, normal))
        turb = np.array(self.bc.turbulence)
        wall = np.asarray(self.bc.kind) == inc.WALL
        turb[wall, 0] = 0.0
        turb[wall, 1] = sst.wall_omega(nu, d1[wall])
        self.bc = self.bc._replace(turbulence=jnp.asarray(turb))

    def _set_transition_inflow(self, conditions):
        """Langtry-Menter inflow values: gamma = 1, Re_thetat from the face's Tu.

        Tu = 100 sqrt(2k/3) / U with the boundary's k and inflow speed (the
        TMR freestream value of Re_thetat, no pressure-gradient factor).
        """
        turb = np.asarray(self.bc.turbulence)
        extra = np.zeros((turb.shape[0], 2))
        extra[:, 0] = 1.0
        for name, bc in conditions.items():
            sl = self.disc.boundary_slices[name]
            tu = lm.turbulence_intensity(turb[sl, 0], _inflow_speed(bc))
            extra[sl, 1] = np.asarray(lm.freestream_re_theta(tu))
        self.bc = self.bc._replace(
            turbulence=jnp.asarray(np.concatenate((turb, extra), axis=1))
        )

    def freestream_transition(self, k, speed):
        """(Tu in percent, Re_thetat) of the Langtry-Menter model for k and U."""
        tu = float(lm.turbulence_intensity(k, speed))
        return tu, float(lm.freestream_re_theta(tu))

    def _production_switch(self, mesh, conditions, regions):
        """1 where SA production is active, 0 in the laminar regions (D15).

        A region's optional ``ramp`` (length) blends the switch linearly over
        that distance inside each x/y bound instead of cutting it off, which
        keeps the SA source differentiable across the transition front.
        """
        switch = np.ones(mesh.n_cells)
        face_boundary = np.full(mesh.n_faces, "", dtype=object)
        for name, faces in mesh.boundary_faces.items():
            face_boundary[faces] = name
        nearest = self.nearest_wall_face
        valid = nearest >= 0
        x, y = mesh.cell_centroid.T
        for region in regions:
            if region["boundary"] not in conditions:
                raise ValueError(
                    f"laminar region: unknown boundary '{region['boundary']}'"
                )
            sel = valid & (face_boundary[np.maximum(nearest, 0)] == region["boundary"])
            side = region.get("side")
            if side is not None:
                ny = mesh.face_normal[np.maximum(nearest, 0), 1]
                # Upper surface: the fluid lies above the wall, the domain
                # normal points down.
                sel &= (ny < 0.0) if side == "upper" else (ny > 0.0)
            ramp = float(region.get("ramp") or 0.0)
            inside = np.ones(mesh.n_cells)  # 1 fully inside the bounds, 0 outside
            for key, arr, sign in (
                ("x_min", x, 1.0),
                ("x_max", x, -1.0),
                ("y_min", y, 1.0),
                ("y_max", y, -1.0),
            ):
                if region.get(key) is not None:
                    margin = sign * (arr - region[key])  # > 0 inside the bound
                    if ramp > 0.0:
                        inside *= np.clip(margin / ramp, 0.0, 1.0)
                    else:
                        inside *= margin >= 0.0
            switch[sel] = np.minimum(switch[sel], 1.0 - inside[sel])
        return switch

    # -- setup checks -------------------------------------------------------

    def check_mass_balance(self, tolerance=1e-10):
        """Reject a prescribed net inflow without a pressure boundary (D39 R3)."""
        if any(bc["type"] in PRESSURE_FIXING for bc in self.conditions.values()):
            return
        v = np.asarray(self.bc.value)[:, 1:3]
        n = np.asarray(self.geom.b_normal)
        flux = np.sum(np.einsum("fk,fk->f", v, n) * np.asarray(self.geom.b_length))
        scale = np.sum(
            np.abs(np.einsum("fk,fk->f", v, n)) * np.asarray(self.geom.b_length)
        )
        if scale > 0.0 and abs(flux) > tolerance * scale:
            raise ValueError(
                f"net inflow {flux:.3e} through velocity boundaries, but no boundary "
                "fixes the pressure level (mass cannot be absorbed)"
            )

    # -- solver interface ---------------------------------------------------

    def scheme_parameters(self, w):
        """Scheme parameters of the state ``w`` (beta^2 for INC, 0 otherwise)."""
        return self.flow.scheme_parameters(self.model, w, self.bc)

    # Older name (INC): the artificial compressibility beta^2.
    beta2 = scheme_parameters

    def parameter_tracker(self):
        """Return ``at(w)``: scheme parameters of ``w``, cached for the last array.

        The residual takes the parameters (beta^2) of the state it is
        evaluated at, including rejected trial states. The Jacobian, the
        pseudo-time term and the Jacobian-vector products belong to the
        accepted state, so they must use its parameters, not the ones a
        rejected trial left behind. The solver passes the same array object
        to all three during one iteration.
        """
        cache = {"w": None, "params": None}

        def at(w):
            if w is not cache["w"]:
                cache["w"], cache["params"] = w, self.scheme_parameters(w)
            return cache["params"]

        return at

    def residual(self, w, params, order=None, psi=None, gamma=None):
        """R(w) for the scheme parameters, spatial order and frozen limiter.

        ``gamma`` is a given transition intermittency (see :meth:`solve`).
        """
        order = self.model.spatial_order if order is None else order
        extra = {} if gamma is None else {"gamma": gamma}
        return np.asarray(
            self.physics.residual(
                self.model,
                self.geom,
                self.bc,
                jnp.asarray(w),
                order,
                params,
                psi,
                **extra,
            )
        )

    def jvp(self, w, dw, params, psi=None, gamma=None):
        """Exact dR/dw @ dw of the second-order residual (``jax.jvp``)."""
        extra = {} if gamma is None else {"gamma": gamma}
        return np.asarray(
            self.physics.residual_jvp(
                self.model,
                self.geom,
                self.bc,
                jnp.asarray(w),
                jnp.asarray(dw),
                self.model.spatial_order,
                params,
                psi,
                **extra,
            )
        )

    def linearize(self, w, params, psi=None, gamma=None):
        """Return ``apply(dw) -> dR/dw(w) @ dw`` linearized once at ``w``.

        ``jax.linearize`` stores the intermediates of the residual and the
        returned function evaluates only the tangent, which costs about a
        quarter of a full ``jax.jvp`` product (C1). The stored arrays live as
        long as the returned function does.
        """
        extra = {} if gamma is None else {"gamma": gamma}
        _, lin = self.physics.residual_linearize(
            self.model,
            self.geom,
            self.bc,
            jnp.asarray(w),
            self.model.spatial_order,
            params,
            psi,
            **extra,
        )

        def apply(dw):
            return np.asarray(_apply_linear(lin, jnp.asarray(dw)))

        return apply

    def limiter(self, w):
        """Limiter values of the state ``w``, shape (n_cells, n_var)."""
        return self.physics.limiter(self.model, self.geom, self.bc, jnp.asarray(w))

    def jacobian(self, w, params):
        """First-order Jacobian, with the pinned row replaced if any.

        With a turbulence model the cell-local source Jacobian is added and,
        for ``coupling = "segregated"``, the mean-flow/turbulence coupling
        blocks are dropped (D34).
        """
        w = jnp.asarray(w)
        b_args = (self.bc.kind, self.bc.value)
        f_args = ()
        mask = None
        if self.model.turbulence == "sst":
            mu_f, dc_f, mu_b, dc_b = rans_sst.jacobian_coefficients(
                self.model, self.geom, self.bc, w
            )
            b_args = b_args + (self.bc.turbulence, mu_b, dc_b)
            f_args = (mu_f, dc_f)
        elif self.turbulent:
            b_args = b_args + (self.bc.nu_tilde,)
        if self.turbulent:
            mask = (
                self.physics.segregated_mask(self.model.n_var)
                if self.model.coupling == "segregated"
                else None
            )
        src = None
        if self.turbulent:
            src = np.asarray(
                self.physics.source_blocks(self.model, self.geom, self.bc, w)
            )
        pin = ()
        if self.model.pin is not None:
            pin = (self.model.pin[0] * self.model.n_var + inc.P,)
        if self._pattern is None:
            self._pattern = self._structural_pattern(
                w, params, b_args, f_args, mask, src
            )
        return first_order_jacobian(
            self.geom,
            self._fi,
            self._fb,
            w,
            params=params,
            b_args=b_args,
            block_mask=mask,
            pattern=self._pattern,
            f_args=f_args,
            diag_blocks=src,
            pin_rows=pin,
        )

    def _structural_pattern(self, w, params, b_args, f_args, mask, src):
        """Jacobian pattern on the structurally non-zero block entries (C2).

        The structure is the union of the non-zero entries over the current
        state and two random states (flow variables shifted, positive
        turbulence variables scaled), so that upwind branches and wall values
        do not hide an entry; an entry that still turns up later widens the
        pattern (see :func:`first_order_jacobian`). The diagonal blocks cover
        the sources and the pseudo-time term.
        """
        rng = np.random.default_rng(12345)
        w0 = np.asarray(w)
        states = [w0]
        # Mean-flow columns shifted, except the positive ones (compressible
        # p and T), which are scaled like the turbulence variables.
        shifted = [c for c in range(3) if c not in self.flow.POSITIVE_COLUMNS]
        for _ in range(2):
            x = w0 * (1.0 + 0.3 * rng.uniform(-1.0, 1.0, w0.shape))
            x[:, shifted] = w0[:, shifted] + 0.3 * rng.standard_normal(
                w0[:, shifted].shape
            ) * (np.abs(w0[:, shifted]).max(axis=0) + 1.0e-3)
            states.append(x)
        diag = []
        for x in states:
            xj = jnp.asarray(x)
            diag.append(
                np.asarray(
                    self.physics.time_term_blocks(
                        self.model, self.geom, xj, params, 10.0
                    )
                )
            )
            if self.turbulent:
                diag.append(
                    np.asarray(
                        self.physics.source_blocks(self.model, self.geom, self.bc, xj)
                    )
                )
        masks = structural_masks(
            self.geom,
            self._fi,
            self._fb,
            [jnp.asarray(x) for x in states],
            params,
            b_args,
            f_args,
            block_mask=mask,
            diag_samples=diag,
        )
        return JacobianPattern(self.geom, self.model.n_var, masks)

    def time_term(self, w, params, cfl, transition_cfl_max=None):
        """Block-diagonal pseudo-time matrix Gamma_i Lambda_i / CFL.

        With ``transition_cfl_max`` (Langtry-Menter) the gamma and Re_thetat
        rows use min(CFL, transition_cfl_max): a separate, lower pseudo-time
        step for the transition equations, as their own solver has in SU2.
        """
        blocks = np.asarray(
            self.physics.time_term_blocks(self.model, self.geom, w, params, cfl)
        )
        if (
            transition_cfl_max is not None
            and self.model.transition == "lm"
            and cfl > transition_cfl_max
        ):
            rows = [rans_sst.NGL, rans_sst.NRT]
            blocks = blocks.copy()
            blocks[:, rows, :] *= cfl / transition_cfl_max
        return block_diagonal(blocks)

    def explicit_update(self, w, r, params, cfl):
        """Explicit increment CFL / Lambda_i * Gamma_i^-1 R_i."""
        if self.turbulent:
            raise NotImplementedError("explicit mode is not available with RANS")
        return np.asarray(
            self.flow.explicit_update(
                self.model, self.geom, jnp.asarray(w), jnp.asarray(r), params, cfl
            )
        )

    # -- loads --------------------------------------------------------------

    def wall_loads(self, w):
        """Pressure and viscous traction on every boundary face."""
        return fz.wall_loads(
            self.model,
            self.geom,
            self.bc,
            jnp.asarray(w),
            self.physics.boundary_state,
            getattr(self.physics, "wall_traction", None),
        )

    def wall_heat_flux(self, w):
        """Wall heat flux into the fluid per boundary face (compressible only)."""
        return np.asarray(
            self.physics.wall_heat_flux(self.model, self.geom, self.bc, jnp.asarray(w))
        )

    def vorticity(self, w):
        """Signed vorticity dv/dx - du/dy per cell (non-dimensional)."""
        w = jnp.asarray(w)
        w_b = self.physics.boundary_state(
            self.bc, self.geom.b_normal, w[self.geom.b_owner]
        )
        grad = lsq_gradient(self.geom, w, w_b)
        return np.asarray(grad[:, 2, 0] - grad[:, 1, 1])

    def intermittency(self, w):
        """Transition intermittency per cell (ones without a transition model)."""
        if not self.turbulent or self.model.transition == "none":
            return np.ones(self.mesh.n_cells)
        return np.asarray(
            self.physics.intermittency_field(
                self.model, self.geom, self.bc, jnp.asarray(w)
            )
        )

    def effective_intermittency(self, w):
        """Langtry-Menter gamma_eff = max(gamma, gamma_sep) per cell."""
        return np.asarray(
            rans_sst.effective_intermittency_field(
                self.model, self.geom, self.bc, jnp.asarray(w)
            )
        )

    def wall_eddy_viscosity_ratio(self, w, faces):
        """Maximum nu_t/nu over the boundary-layer column above each wall face.

        The column is the set of cells whose nearest wall face it is, within
        0.05 reference lengths of the wall (several boundary-layer
        thicknesses at any Reynolds number of interest, but not the
        freestream). A laminar boundary layer has values of 1e-6-1e-2, a
        turbulent one 10-100; the transition location reported in the
        summary is where it first exceeds 1. ``faces`` are boundary-face
        indices (``b_*`` order).
        """
        ratio = self.eddy_viscosity(w) / self.kinematic_viscosity(w)
        face_ids = self.disc.boundary_face_ids[np.asarray(faces)]
        nearest = self.nearest_wall_face
        column = np.full(self.mesh.n_faces, -np.inf)
        valid = (nearest >= 0) & (np.asarray(self.geom.wall_distance) <= 0.05)
        np.maximum.at(column, nearest[valid], ratio[valid])
        return np.where(np.isfinite(column[face_ids]), column[face_ids], 0.0)

    def kinematic_viscosity(self, w):
        """Molecular kinematic viscosity per cell (compressible: mu(T) / rho)."""
        if self.compressible:
            return np.asarray(
                rans_compressible.kinematic_viscosity(self.model, jnp.asarray(w))
            )
        return np.full(w.shape[0], self.model.viscosity / self.model.density)

    def wall_y_plus(self, w, faces, tau_w):
        """y+ = d sqrt(tau_w / rho_w) / nu_w of the wall cells of ``faces``.

        ``tau_w`` (non-dimensional) per face; rho_w and nu_w = mu(T_w) / rho_w
        of the wall boundary state (constants for incompressible flow).
        """
        faces = np.asarray(faces)
        owner = np.asarray(self.geom.b_owner)[faces]
        dist = np.asarray(self.geom.wall_distance)[owner]
        if self.compressible:
            w_b = np.asarray(
                self.physics.boundary_state(
                    self.bc, self.geom.b_normal, jnp.asarray(w)[self.geom.b_owner]
                )
            )[faces]
            rho = np.asarray(comp.density(self.model.gas_constant, w_b))
            nu = np.asarray(rans_compressible.kinematic_viscosity(self.model, w_b))
        else:
            rho = self.model.density
            nu = self.model.viscosity / self.model.density
        return dist * np.sqrt(np.abs(tau_w) / rho) / nu

    def eddy_viscosity(self, w):
        """Kinematic eddy viscosity per cell (zeros without a turbulence model)."""
        if not self.turbulent:
            return np.zeros(self.mesh.n_cells)
        if self.model.turbulence == "sst":
            return np.asarray(
                rans_sst.eddy_viscosity(self.model, self.geom, self.bc, jnp.asarray(w))
            )
        return np.asarray(self.physics.eddy_viscosity(self.model, jnp.asarray(w)))

    def blending_f1(self, w):
        """Menter's first SST blending function per cell (SST only)."""
        return np.asarray(
            rans_sst.blending_f1(self.model, self.geom, self.bc, jnp.asarray(w))
        )

    def boundary_faces(self, names):
        """Boundary-face indices (``b_*`` order) of the named boundaries."""
        idx = [
            np.arange(
                self.disc.boundary_slices[n].start, self.disc.boundary_slices[n].stop
            )
            for n in names
        ]
        return np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)

    def forces(self, w, boundaries=None, moment_center=(0.0, 0.0), loads=None):
        """Integrated force and moment on the given boundaries (default: all walls).

        Returns
        -------
        pyaeroflow.physics.forces.Forces
        """
        if boundaries is None:
            boundaries = [
                n
                for n, bc in self.conditions.items()
                if bc["type"] in ("wall", "transpiration-wall")
            ]
        loads = self.wall_loads(w) if loads is None else loads
        return fz.integrate(
            loads, self.geom, self.boundary_faces(boundaries), moment_center
        )

    # -- farfield point-vortex correction (P10) --------------------------------

    def farfield_vortex(self, centre, boundaries=None, relaxation=0.5):
        """Return a :class:`FarfieldVortex` for this problem.

        The farfield state becomes freestream plus the velocity induced by a
        point vortex at ``centre`` whose circulation follows from the lift on
        ``boundaries`` (Kutta-Joukowski, Gamma = L / (rho U)), with the
        pressure from Bernoulli. NASA TMR recommends it for farfields closer
        than about 500 chords.
        """
        faces = self.boundary_faces(
            [n for n, bc in self.conditions.items() if bc["type"] == "farfield"]
        )
        if faces.size == 0:
            raise ValueError("the vortex correction needs a farfield boundary")
        if self.compressible:
            raise ValueError("the vortex correction is incompressible only")
        if boundaries is None:
            boundaries = [
                n
                for n, bc in self.conditions.items()
                if bc["type"] in ("wall", "transpiration-wall")
            ]
        return FarfieldVortex(
            faces,
            np.asarray(self.bc.value)[faces].copy(),
            np.asarray(self.geom.b_centre)[faces] - np.asarray(centre, dtype=float),
            list(boundaries),
            float(relaxation),
        )

    def apply_farfield_vortex(self, vortex, w):
        """Update the farfield values from the current lift (one relaxed step)."""
        base = vortex.base
        u_inf = base[:, 1:3]
        speed = np.hypot(u_inf[:, 0], u_inf[:, 1]).mean()
        f = self.forces(w, vortex.boundaries)
        direction = u_inf.mean(axis=0) / max(speed, 1e-300)
        lift = -f.fx * direction[1] + f.fy * direction[0]
        gamma = lift / (self.model.density * speed)
        vortex.gamma += vortex.relaxation * (gamma - vortex.gamma)
        self._set_vortex_values(vortex)

    def restore_farfield_vortex(self, vortex, gamma):
        """Set the circulation (e.g. from a restart file) and the farfield values."""
        vortex.gamma = float(gamma)
        self._set_vortex_values(vortex)

    def _set_vortex_values(self, vortex):
        """Farfield p, u, v of freestream plus the vortex of ``vortex.gamma``."""
        base = vortex.base
        u_inf = base[:, 1:3]
        speed = np.hypot(u_inf[:, 0], u_inf[:, 1]).mean()
        r = vortex.offset
        r2 = np.sum(r**2, axis=1)
        coeff = vortex.gamma / (2.0 * np.pi * r2)
        vel = u_inf + coeff[:, None] * np.column_stack((r[:, 1], -r[:, 0]))
        p = base[:, 0] + 0.5 * self.model.density * (speed**2 - np.sum(vel**2, axis=1))
        value = np.asarray(self.bc.value).copy()
        value[vortex.faces, 0] = p
        value[vortex.faces, 1:3] = vel
        self.bc = self.bc._replace(value=jnp.asarray(value))

    def initial_state(
        self,
        velocity=(0.0, 0.0),
        pressure=0.0,
        nu_tilde=None,
        k_omega=None,
        temperature=1.0,
    ):
        """Uniform initial field, shape (n_cells, n_var); nu_tilde default 3 nu.

        Compressible flow: (p, u, v, T) with the given ``temperature``.

        The SA-AFT columns start at gamma_tilde = 0 and n = 0 (freestream).
        SST takes ``k_omega`` = (k, omega); default Tu = 0.1 % of the given
        velocity (or 1) and nu_t / nu = 1. Langtry-Menter starts at gamma = 1
        and the freestream Re_thetat of that k and velocity.
        """
        w = np.zeros((self.mesh.n_cells, self.model.n_var))
        w[:, inc.P] = pressure
        w[:, inc.U] = velocity[0]
        w[:, inc.V] = velocity[1]
        if self.compressible:
            w[:, comp.T] = temperature
        if self.turbulent:
            nu = self.model.viscosity / self.model.density
            if self.model.turbulence == "sst":
                if k_omega is None:
                    speed = float(np.hypot(*velocity)) or 1.0
                    k_omega = sst_default_values(nu, speed)
                w[:, rans_sst.NK] = k_omega[0]
                w[:, rans_sst.NW] = k_omega[1]
                if self.model.transition == "lm":
                    speed = float(np.hypot(*velocity)) or 1.0
                    w[:, rans_sst.NGL] = 1.0
                    w[:, rans_sst.NRT] = self.freestream_transition(k_omega[0], speed)[
                        1
                    ]
            else:
                nt = 3.0 * nu if nu_tilde is None else nu_tilde
                w[:, self.turbulence_column] = nt
        return w

    def solve(
        self,
        w0=None,
        settings=None,
        callback=None,
        freeze_limiter_after_iterations=None,
        freeze_limiter_after_residual_drop=None,
        vortex=None,
        release_limiter=False,
        gamma_relaxation=None,
        gamma_diagnostics=False,
        nu_tilde_step_limit=None,
        transition_clip=True,
        gamma_guard=False,
        resume=None,
        debug=None,
        transition_cfl_max=None,
        whole_cell_relaxation=False,
    ):
        """Implicit pseudo-time solution (D25, D35).

        beta^2 is updated from the current state before every residual
        evaluation, as in SU2. The limiter is frozen (D23) once either freeze
        condition is met; afterwards the residual, its Jacobian-vector
        products and the solution use the limiter values of that state.
        With ``vortex`` (:meth:`farfield_vortex`) the farfield values are
        updated from the lift after every accepted iteration (P10). With
        ``release_limiter`` the freeze sets the limiter to one instead
        (unlimited second order after a limited start-up): for smooth
        high-Reynolds-number flows the limiter only costs accuracy in the
        boundary layer once the transient has passed.

        With ``gamma_relaxation`` = omega < 1 (transition model) the
        intermittency in the residual is a lagged field, advanced after every
        accepted iteration by gamma += omega (gamma_model(w) - gamma): a cell
        crossing the onset threshold ramps its production on over about
        1 / omega iterations instead of in one step. The converged state is
        that of the unrelaxed model. ``gamma_diagnostics`` stores per-iteration
        intermittency statistics (:data:`GAMMA_DIAGNOSTICS`) in the callback's
        ``info["transition_diagnostics"]``.

        ``debug`` is passed to :func:`solve_steady` (numerics record of every
        iteration, see :mod:`pyaeroflow.io.debug_log`).

        ``transition_cfl_max`` caps the pseudo-time CFL of the Langtry-Menter
        gamma and Re_thetat rows (:meth:`time_term`); None = solver CFL.
        ``whole_cell_relaxation`` (SST): a cell whose k-omega update the
        positivity limiter scales gets the same scale on all its variables
        (:meth:`_sst_limiter`).

        With ``nu_tilde_step_limit`` = f every update of nu_tilde is capped
        to f max(|nu_tilde|, nu) per cell before the line search (the
        mean-flow update is untouched), as SU2 limits its turbulence
        variable: a Newton step that multiplies nu_tilde in one boundary-layer
        cell cannot pass as a small change of the overall residual norm.

        With the Langtry-Menter model the update clips gamma and Re_thetat to
        SU2's limits (:meth:`_sst_limiter`); ``transition_clip=False`` leaves
        them unclipped. ``gamma_guard`` (off by default) replaces a Newton
        update of gamma that the clip floor would discard while the residual
        asks for the other direction by the explicit local step
        (:meth:`_sst_limiter`); the counts go to the callback's
        ``info["gamma_guard"]``. The positive linearization of the gamma
        source is a model flag (``lm_positive_jacobian``).

        ``resume`` is the state of an earlier run (the ``state`` of its result
        or of a restart file, see :func:`pyaeroflow.run.load_restart_state`):
        the solver state of :func:`solve_steady` plus the frozen or released
        limiter ``psi``, the lagged intermittency ``gamma`` and the farfield
        circulation ``vortex_gamma``. The result's ``state`` is the same
        dictionary at the end of the run.
        """
        w0 = self.initial_state() if w0 is None else w0
        resume = resume or {}
        state = {
            "params": self.scheme_parameters(w0),
            "psi": None,
            "gamma": None,
            "model": None,
        }
        if resume.get("psi") is not None:
            state["psi"] = jnp.asarray(resume["psi"])
        if vortex is not None and resume.get("vortex_gamma") is not None:
            self.restore_farfield_vortex(vortex, resume["vortex_gamma"])
        relax = (
            gamma_relaxation is not None
            and gamma_relaxation < 1.0
            and self.model.transition == "bcm"
        )
        track = relax or (gamma_diagnostics and self.model.transition != "none")
        if track:
            state["model"] = self.physics.intermittency_field(
                self.model, self.geom, self.bc, jnp.asarray(w0)
            )
            if relax:
                state["gamma"] = state["model"]
                if resume.get("gamma") is not None:
                    state["gamma"] = jnp.asarray(resume["gamma"])
        uses_limiter = self.model.limiter != "none" and self.model.spatial_order == 2
        params_at = self.parameter_tracker()

        def residual(w):
            state["params"] = self.scheme_parameters(w)
            return self.residual(
                w, state["params"], psi=state["psi"], gamma=state["gamma"]
            )

        def residual_startup(w):
            state["params"] = self.scheme_parameters(w)
            return self.residual(w, state["params"], order=1, gamma=state["gamma"])

        def transition_step(it, w):
            """Advance the lagged intermittency, record the diagnostics."""
            g_model = self.physics.intermittency_field(
                self.model, self.geom, self.bc, jnp.asarray(w)
            )
            change = np.abs(np.asarray(g_model - state["model"]))
            state["model"] = g_model
            if relax and it > 0:
                state["gamma"] = state["gamma"] + gamma_relaxation * (
                    g_model - state["gamma"]
                )
                on_iteration.info["residual_changed"] = True
            if gamma_diagnostics:
                on_iteration.info["transition_diagnostics"] = self._gamma_diagnostics(
                    w, g_model, change, state
                )

        def on_iteration(it, w, residuals, cfl, accepted):
            if vortex is not None and accepted:
                self.apply_farfield_vortex(vortex, w)
                # The farfield values changed: the residual operator did, so
                # solve_steady must judge the next step against the residual
                # of the current state under the new farfield. Without this the
                # reference is the old operator's residual: every trial step
                # looks worse, the line search cuts it to a quarter, and with
                # a small reject_factor even a vanishing step is rejected.
                on_iteration.info["residual_changed"] = True
            if track and accepted:
                transition_step(it, w)
            if uses_limiter and state["psi"] is None and accepted:
                # Orders of the per-equation convergence measure (D54), the
                # same quantity that decides the Newton-Krylov switch.
                measure = on_iteration.info.get("measure", 1.0)
                drop = -np.log10(max(measure, np.finfo(float).tiny))
                if (
                    freeze_limiter_after_iterations is not None
                    and it >= freeze_limiter_after_iterations
                ) or (
                    freeze_limiter_after_residual_drop is not None
                    and drop >= freeze_limiter_after_residual_drop
                ):
                    state["psi"] = (
                        jnp.ones_like(jnp.asarray(w))
                        if release_limiter
                        else self.limiter(w)
                    )
                    # The residual operator changed: solve_steady re-evaluates
                    # its reference residual before judging the next step.
                    on_iteration.info["residual_changed"] = True
                    if callback is not None and hasattr(callback, "info"):
                        callback.info["note"] = (
                            "limiter released" if release_limiter else "limiter frozen"
                        )
            if callback is not None:
                callback(it, w, residuals, cfl, accepted)

        on_iteration.info = getattr(callback, "info", {})
        if callback is not None and not hasattr(callback, "info"):
            callback.info = on_iteration.info
        guard_stats = None
        if self.model.transition == "lm" and gamma_guard:
            guard_stats = {"updates": 0, "fired": 0, "cells_max": 0, "cells_last": 0}
            on_iteration.info["gamma_guard"] = guard_stats

        def driver_state():
            """Limiter, lagged intermittency and vortex of the live run."""
            out = {}
            if state["psi"] is not None:
                out["psi"] = np.asarray(state["psi"])
            if state["gamma"] is not None:
                out["gamma"] = np.asarray(state["gamma"])
            if vortex is not None:
                out["vortex_gamma"] = vortex.gamma
            return out

        on_iteration.info["driver_state"] = driver_state

        result = solve_steady(
            residual,
            lambda w: self.jacobian(w, params_at(w)),
            lambda w, cfl: self.time_term(w, params_at(w), cfl, transition_cfl_max),
            w0,
            settings or PseudoTimeSettings(),
            residual_startup=residual_startup,
            callback=on_iteration,
            jvp=lambda w, dw: self.jvp(
                w, dw, params_at(w), state["psi"], state["gamma"]
            ),
            linearize=lambda w: self.linearize(
                w, params_at(w), state["psi"], state["gamma"]
            ),
            limit_step=self._step_limiter(
                nu_tilde_step_limit,
                transition_clip,
                gamma_guard,
                guard_stats,
                whole_cell_relaxation,
            ),
            resume=resume,
            debug=debug,
        )
        return result._replace(state={**result.state, **driver_state()})

    def _step_limiter(
        self, factor, clip=True, gamma_guard=False, guard_stats=None, whole_cell=False
    ):
        """Step limiter of the turbulence variables, or None.

        SST always gets SU2's under-relaxation (:meth:`_sst_limiter`, with the
        clip of the transition variables unless ``clip`` is False, and the
        descent guard of gamma unless ``gamma_guard`` is False); SA the
        nu_tilde cap of ``factor`` (None = no cap).
        """
        if self.compressible:
            positivity = self._positivity_limiter()
            cap = self._nu_tilde_limiter(factor)
            if cap is None:
                return positivity

            def limit_step(u, du, r=None, tau=None):
                return positivity(u, cap(u, du), r, tau)

            return limit_step
        if self.model.turbulence == "sst":
            return self._sst_limiter(
                clip=clip, guard=gamma_guard, stats=guard_stats, whole_cell=whole_cell
            )
        return self._nu_tilde_limiter(factor)

    def _positivity_limiter(self, max_drop=COMPRESSIBLE_MAX_DROP):
        """Compressible flow: p and T stay positive.

        Per cell the whole update is scaled by min(1, max_drop / ratio), with
        ratio the larger relative decrease -dp / p, -dT / T; increases are left
        to the line search. With ``max_drop`` < 1 a full step keeps at least
        1 - max_drop of the old values, so that no trial state of the line
        search has a negative pressure or temperature.
        """
        cols = list(self.flow.POSITIVE_COLUMNS)

        def limit_step(u, du, r=None, tau=None):
            val = np.maximum(u[:, cols], 1e-300)
            drop = (np.maximum(-du[:, cols], 0.0) / val).max(axis=1)
            scale = np.minimum(1.0, max_drop / np.maximum(drop, 1e-300))
            return du * scale[:, None]

        return limit_step

    def _sst_limiter(
        self,
        max_update=SST_MAX_UPDATE,
        clip=True,
        guard=False,
        stats=None,
        whole_cell=False,
    ):
        """SU2's SST under-relaxation: k and omega stay positive.

        After the update k and omega are clipped to SU2's limits
        (:data:`SST_K_LIMITS`, :data:`SST_OMEGA_LIMITS`). Without the floor
        the 99 % cut lets k fall by a factor 100 per iteration in a cell where
        Newton keeps asking for a negative k: on the 320k SD7003 LM alpha 0 a
        wake cell reached k = 3e-73 with u = -0.23 among neighbours at u =
        0.93, its Jacobian (sqrt(k) terms) was useless, and the cell held 99 %
        of the u residual and cut every later step.

        Per cell the (k, omega) update is scaled by min(1, max_update /
        ratio) with ratio the larger of |dk| / k and |domega| / omega, only
        where the update lowers a variable (SU2 scales on any change; growth
        is left to the line search here, so that the turbulence can develop
        from a low freestream). With Langtry-Menter gamma and Re_thetat are
        included, and after the update they are clipped as in SU2
        (``CTransLMSolver``): gamma to [1e-4, 5], Re_thetat to [1e-4, 1e15].
        Without the floor a cell whose update keeps asking for a negative gamma
        loses 99 % per iteration (gamma = 1e-142 after 70 iterations) and the
        square root of the onset term has an enormous derivative there.
        ``clip=False`` leaves the transition variables unclipped.

        Descent guard (``guard``, Langtry-Menter): a cell whose Newton update
        of gamma would end below the clip floor while its residual has the
        same sign as the update (R < 0 asks for a larger gamma, the update
        lowers it) gets the explicit local step -R / tau instead, tau the
        pseudo-time diagonal at CFL 1. At the floor the sqrt(gamma) of the
        production has the derivative 1 / (2 sqrt gamma) = 50, which turns the
        diagonal of the gamma equation negative in a cell of the shear layer;
        the Newton update then points away from the root, the clip discards
        it, and the cell stays at the floor with a residual of hundreds
        (320k SD7003: 8 such cells held the gamma residual at 10^-5.4). The
        limiter needs ``r`` and ``tau`` for the guard (:func:`solve_steady`
        passes them); with ``r=None`` the guard is skipped. ``stats`` (dict)
        counts the updates seen, those where the guard fired, the largest and
        the last number of cells.

        ``whole_cell``: the scale of the k-omega group also multiplies the
        flow (and transition) update of that cell. Scaling only k and omega
        leaves a flow update that was computed for the full k-omega change, so
        in a cell where Newton keeps asking for a negative k or omega the
        coupled update no longer fits together and can raise the residual
        at any step length.
        """
        groups = [[rans_sst.NK, rans_sst.NW]]
        clips = [(rans_sst.NK, SST_K_LIMITS), (rans_sst.NW, SST_OMEGA_LIMITS)]
        guard = guard and self.model.transition == "lm"
        if self.model.transition == "lm":
            groups.append(
                [rans_sst.NGL, rans_sst.NRT]
            )  # scaled apart (own solver in SU2)
            if clip:
                clips += [
                    (rans_sst.NGL, lm.GAMMA_LIMITS),
                    (rans_sst.NRT, lm.RE_THETA_LIMITS),
                ]

        def limit_step(u, du, r=None, tau=None):
            du = du.copy()
            if guard and r is not None:
                col = rans_sst.NGL
                g, dg, rg = u[:, col], du[:, col], r[:, col]
                wrong = (dg * rg > 0.0) & (g + dg < lm.GAMMA_LIMITS[0])
                n_wrong = int(np.count_nonzero(wrong))
                if n_wrong:
                    du[wrong, col] = -rg[wrong] / tau[wrong, col]
                if stats is not None:
                    stats["updates"] += 1
                    stats["fired"] += n_wrong > 0
                    stats["cells_max"] = max(stats["cells_max"], n_wrong)
                    stats["cells_last"] = n_wrong
            for i, cols in enumerate(groups):
                val = np.maximum(u[:, cols], 1e-300)
                drop = np.maximum(-du[:, cols], 0.0) / val
                ratio = drop.max(axis=1)
                scale = np.minimum(1.0, max_update / np.maximum(ratio, 1e-300))
                if whole_cell and i == 0:
                    du *= scale[:, None]  # k-omega group: the whole cell
                else:
                    du[:, cols] *= scale[:, None]
            for col, (low, high) in clips:
                du[:, col] = np.clip(u[:, col] + du[:, col], low, high) - u[:, col]
            return du

        return limit_step

    def _nu_tilde_limiter(self, factor):
        """Step limiter capping the nu_tilde update to factor max(|nu_tilde|, nu).

        Returns ``limit_step(u, du)``, or None without a cap or turbulence.
        """
        if factor is None or not self.turbulent:
            return None
        nu_floor = self.model.viscosity / self.model.density
        col = self.turbulence_column

        def limit_step(u, du, r=None, tau=None):
            cap = factor * np.maximum(np.abs(u[:, col]), nu_floor)
            du = du.copy()
            du[:, col] = np.clip(du[:, col], -cap, cap)
            return du

        return limit_step

    def _gamma_statistics(self, g_model, change):
        """Return the first six :data:`GAMMA_DIAGNOSTICS` (near-wall cells only)."""
        c, area = self.mesh.cell_centroid, self.mesh.cell_area
        near = np.asarray(self.geom.wall_distance) < GAMMA_DIAGNOSTICS_DISTANCE
        g = np.where(near, np.asarray(g_model), 0.0)
        change = np.where(near, change, 0.0)
        i = int(np.argmax(change))
        return (
            int(np.count_nonzero(g > 0.01)),
            float(np.sum(g * area)),
            int(np.count_nonzero(change > 0.05)),
            float(change[i]),
            float(c[i, 0]),
            float(c[i, 1]),
        )

    def gamma_step_diagnostics(self, w, previous=None):
        """Per-time-step intermittency statistics of an unsteady run.

        Returns (values of :data:`GAMMA_STEP_DIAGNOSTICS`, model gamma of
        ``w``); ``previous`` is the model gamma of the previous step (``None``:
        no change is counted).
        """
        g = self.physics.intermittency_field(
            self.model, self.geom, self.bc, jnp.asarray(w)
        )
        change = (
            np.zeros(len(g)) if previous is None else np.abs(np.asarray(g - previous))
        )
        return self._gamma_statistics(g, change), g

    def _gamma_diagnostics(self, w, g_model, change, state):
        """Values of :data:`GAMMA_DIAGNOSTICS` for the accepted state ``w``."""
        c = self.mesh.cell_centroid
        near = np.asarray(self.geom.wall_distance) < GAMMA_DIAGNOSTICS_DISTANCE
        r_nu = np.where(
            near,
            np.abs(
                self.residual(
                    w, state["params"], psi=state["psi"], gamma=state["gamma"]
                )[:, rans.NT]
            ),
            0.0,
        )
        j = int(np.argmax(r_nu))
        return self._gamma_statistics(g_model, change) + (
            float(r_nu[j]),
            float(c[j, 0]),
            float(c[j, 1]),
        )

    def mass(self):
        """Diagonal of the mass matrix: rho * area on u, v; area on turbulence."""
        if self.compressible:
            raise NotImplementedError("unsteady compressible flow is planned")
        area = np.asarray(self.geom.area)
        rho = self.model.density
        diag = [0.0, rho, rho] + [1.0] * (self.model.n_var - 3)
        return area[:, None] * np.array(diag)

    def solve_unsteady(
        self,
        w0,
        settings,
        callback=None,
        inner_callback=None,
        nu_tilde_step_limit=None,
        transition_clip=True,
        gamma_guard=False,
        resume=None,
    ):
        """Dual time stepping from ``w0`` to ``settings.end_time`` (D9, D37).

        Every time step solves the steady residual plus the BDF time term
        with the inner pseudo-time/Newton-Krylov iterations of
        ``settings.inner``; beta^2 follows the current state, the limiter is
        not frozen. ``nu_tilde_step_limit`` caps every inner update of
        nu_tilde, ``transition_clip`` clips gamma and Re_thetat and
        ``gamma_guard`` guards the gamma update as in :meth:`solve` (the guard
        then uses the inner residual). ``resume`` continues an earlier run
        (``step`` and ``u_previous``, see :func:`solve_unsteady`).
        """
        params_at = self.parameter_tracker()

        def residual(w):
            return self.residual(w, self.scheme_parameters(w))

        return solve_unsteady(
            residual,
            lambda w: self.jacobian(w, params_at(w)),
            lambda w, cfl: self.time_term(w, params_at(w), cfl),
            self.mass(),
            w0,
            settings,
            callback=callback,
            inner_callback=inner_callback,
            jvp=lambda w, dw: self.jvp(w, dw, params_at(w)),
            linearize=lambda w: self.linearize(w, params_at(w)),
            limit_step=self._step_limiter(
                nu_tilde_step_limit, transition_clip, gamma_guard
            ),
            resume=resume,
        )

    def solve_explicit(
        self, w0=None, cfl=0.8, max_iterations=20000, residual_drop=8.0, callback=None
    ):
        """Explicit Runge-Kutta reference mode (D25)."""
        w0 = self.initial_state() if w0 is None else w0
        state = {"params": self.scheme_parameters(w0)}

        def residual(w):
            state["params"] = self.scheme_parameters(w)
            return self.residual(w, state["params"])

        return solve_explicit(
            residual,
            lambda w, r, cfl: self.explicit_update(w, r, state["params"], cfl),
            w0,
            cfl=cfl,
            max_iterations=max_iterations,
            residual_drop=residual_drop,
            callback=callback,
        )


# Older name of :class:`FlowProblem` (the incompressible solver came first).
IncompressibleProblem = FlowProblem
