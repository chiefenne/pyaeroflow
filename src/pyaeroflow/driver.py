"""Set up and solve an incompressible flow problem (Python API, D41).

:class:`IncompressibleProblem` binds a mesh, a model and boundary conditions
and provides the residual, Jacobian and pseudo-time term for the solvers in
:mod:`pyaeroflow.solver`.
"""

from functools import partial

import jax.numpy as jnp
import numpy as np

from pyaeroflow.numerics import build_geometry
from pyaeroflow.numerics.wall_distance import mesh_wall_distance
from pyaeroflow.physics import forces as fz
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import rans
from pyaeroflow.solver import (
    JacobianPattern,
    PseudoTimeSettings,
    block_diagonal,
    first_order_jacobian,
    replace_rows,
    solve_explicit,
    solve_steady,
    solve_unsteady,
)

PRESSURE_FIXING = {"pressure-outlet", "farfield", "total-pressure-inlet"}
INFLOW_TYPES = {"velocity-inlet", "total-pressure-inlet", "farfield"}


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

# Per-iteration transition diagnostics of :meth:`IncompressibleProblem.solve`:
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


class IncompressibleProblem:
    """An incompressible flow case on a mesh.

    Parameters
    ----------
    mesh : Mesh
    model : IncompressibleModel
        If ``model.pin`` is None and no boundary fixes the pressure level
        (pressure outlet or farfield), the pressure of the cell nearest
        ``pressure_point`` is pinned to ``pin_pressure`` (D39 R2).
    boundary_conditions : dict of str to dict
        See :func:`pyaeroflow.physics.incompressible.boundary_data`. Periodic
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
        self.turbulent = model.turbulence != "none"
        if self.turbulent:
            if model.turbulence != "sa-neg":
                raise ValueError(f"unknown turbulence model '{model.turbulence}'")
            if model.viscosity <= 0.0:
                raise ValueError("a turbulence model needs a positive viscosity")
            # Default inflow value nu_tilde = 3 nu (NASA TMR, D33).
            regular = {
                name: (
                    {"nu_tilde": 3.0 * model.viscosity / model.density, **bc}
                    if bc["type"] in INFLOW_TYPES
                    else bc
                )
                for name, bc in regular.items()
            }
        self.disc = build_geometry(
            mesh, weight_exponent, periodic, inc.lsq_include(regular, model.n_var)
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
        self.bc = inc.boundary_data(self.disc, regular, model.density)
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
        self.physics = rans if self.turbulent else inc
        self._pattern = None  # Jacobian sparsity, built at the first assembly
        self._fi = partial(self.physics.interior_flux, model)
        self._fb = partial(self.physics.boundary_flux, model)

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
        v = np.asarray(self.bc.value)[:, 1:]
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

    def beta2(self, w):
        """Artificial compressibility beta^2 for the current state."""
        return inc.beta_squared(self.model, np.asarray(w)[:, :3], self.bc)

    def residual(self, w, beta2, order=None, psi=None, gamma=None):
        """R(w) for the current beta^2, spatial order and frozen limiter.

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
                beta2,
                psi,
                **extra,
            )
        )

    def jvp(self, w, dw, beta2, psi=None, gamma=None):
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
                beta2,
                psi,
                **extra,
            )
        )

    def limiter(self, w):
        """Limiter values of the state ``w``, shape (n_cells, n_var)."""
        return self.physics.limiter(self.model, self.geom, self.bc, jnp.asarray(w))

    def jacobian(self, w, beta2):
        """First-order Jacobian, with the pinned row replaced if any.

        With a turbulence model the cell-local source Jacobian is added and,
        for ``coupling = "segregated"``, the mean-flow/turbulence coupling
        blocks are dropped (D34).
        """
        w = jnp.asarray(w)
        b_args = (self.bc.kind, self.bc.value)
        mask = None
        if self.turbulent:
            b_args = b_args + (self.bc.nu_tilde,)
            mask = rans.SEGREGATED_MASK if self.model.coupling == "segregated" else None
        if self._pattern is None:
            self._pattern = JacobianPattern(self.geom, self.model.n_var)
        j = first_order_jacobian(
            self.geom,
            self._fi,
            self._fb,
            w,
            params=beta2,
            b_args=b_args,
            block_mask=mask,
            pattern=self._pattern,
        )
        if self.turbulent:
            j = j + block_diagonal(
                np.asarray(rans.source_blocks(self.model, self.geom, self.bc, w))
            )
        if self.model.pin is not None:
            j = replace_rows(j, [self.model.pin[0] * self.model.n_var + inc.P])
        return j

    def time_term(self, w, beta2, cfl):
        """Block-diagonal pseudo-time matrix Gamma_i Lambda_i / CFL."""
        return block_diagonal(
            np.asarray(
                self.physics.time_term_blocks(self.model, self.geom, w, beta2, cfl)
            )
        )

    def explicit_update(self, w, r, beta2, cfl):
        """Explicit increment CFL / Lambda_i * Gamma_i^-1 R_i."""
        if self.turbulent:
            raise NotImplementedError("explicit mode is not available with RANS")
        return np.asarray(
            inc.explicit_update(
                self.model, self.geom, jnp.asarray(w), jnp.asarray(r), beta2, cfl
            )
        )

    # -- loads --------------------------------------------------------------

    def wall_loads(self, w):
        """Pressure and viscous traction on every boundary face."""
        return fz.wall_loads(
            self.model, self.geom, self.bc, jnp.asarray(w), self.physics.boundary_state
        )

    def intermittency(self, w):
        """Transition intermittency per cell (ones without a transition model)."""
        if not self.turbulent or self.model.transition == "none":
            return np.ones(self.mesh.n_cells)
        return np.asarray(
            rans.intermittency_field(self.model, self.geom, self.bc, jnp.asarray(w))
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
        ratio = self.eddy_viscosity(w) / (self.model.viscosity / self.model.density)
        face_ids = self.disc.boundary_face_ids[np.asarray(faces)]
        nearest = self.nearest_wall_face
        column = np.full(self.mesh.n_faces, -np.inf)
        valid = (nearest >= 0) & (np.asarray(self.geom.wall_distance) <= 0.05)
        np.maximum.at(column, nearest[valid], ratio[valid])
        return np.where(np.isfinite(column[face_ids]), column[face_ids], 0.0)

    def eddy_viscosity(self, w):
        """Kinematic eddy viscosity per cell (zeros without a turbulence model)."""
        if not self.turbulent:
            return np.zeros(self.mesh.n_cells)
        return np.asarray(rans.eddy_viscosity(self.model, jnp.asarray(w)))

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
        r = vortex.offset
        r2 = np.sum(r**2, axis=1)
        coeff = vortex.gamma / (2.0 * np.pi * r2)
        vel = u_inf + coeff[:, None] * np.column_stack((r[:, 1], -r[:, 0]))
        p = base[:, 0] + 0.5 * self.model.density * (speed**2 - np.sum(vel**2, axis=1))
        value = np.asarray(self.bc.value).copy()
        value[vortex.faces, 0] = p
        value[vortex.faces, 1:3] = vel
        self.bc = self.bc._replace(value=jnp.asarray(value))

    def initial_state(self, velocity=(0.0, 0.0), pressure=0.0, nu_tilde=None):
        """Uniform initial field, shape (n_cells, n_var); nu_tilde default 3 nu."""
        w = np.zeros((self.mesh.n_cells, self.model.n_var))
        w[:, inc.P] = pressure
        w[:, inc.U] = velocity[0]
        w[:, inc.V] = velocity[1]
        if self.turbulent:
            nu = self.model.viscosity / self.model.density
            w[:, rans.NT] = 3.0 * nu if nu_tilde is None else nu_tilde
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

        With ``nu_tilde_step_limit`` = f every update of nu_tilde is capped
        to f max(|nu_tilde|, nu) per cell before the line search (the
        mean-flow update is untouched), as SU2 limits its turbulence
        variable: a Newton step that multiplies nu_tilde in one boundary-layer
        cell cannot pass as a small change of the overall residual norm.
        """
        w0 = self.initial_state() if w0 is None else w0
        state = {"beta2": self.beta2(w0), "psi": None, "gamma": None, "model": None}
        relax = (
            gamma_relaxation is not None
            and gamma_relaxation < 1.0
            and self.model.transition != "none"
        )
        track = relax or (gamma_diagnostics and self.model.transition != "none")
        if track:
            state["model"] = self.physics.intermittency_field(
                self.model, self.geom, self.bc, jnp.asarray(w0)
            )
            if relax:
                state["gamma"] = state["model"]
        uses_limiter = self.model.limiter != "none" and self.model.spatial_order == 2

        def residual(w):
            state["beta2"] = self.beta2(w)
            return self.residual(
                w, state["beta2"], psi=state["psi"], gamma=state["gamma"]
            )

        def residual_startup(w):
            state["beta2"] = self.beta2(w)
            return self.residual(w, state["beta2"], order=1, gamma=state["gamma"])

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

        limit_step = None
        if nu_tilde_step_limit is not None and self.turbulent:
            nu_floor = self.model.viscosity / self.model.density

            def limit_step(u, du):
                cap = nu_tilde_step_limit * np.maximum(np.abs(u[:, rans.NT]), nu_floor)
                du = du.copy()
                du[:, rans.NT] = np.clip(du[:, rans.NT], -cap, cap)
                return du

        return solve_steady(
            residual,
            lambda w: self.jacobian(w, state["beta2"]),
            lambda w, cfl: self.time_term(w, state["beta2"], cfl),
            w0,
            settings or PseudoTimeSettings(),
            residual_startup=residual_startup,
            callback=on_iteration,
            jvp=lambda w, dw: self.jvp(
                w, dw, state["beta2"], state["psi"], state["gamma"]
            ),
            limit_step=limit_step,
        )

    def _gamma_diagnostics(self, w, g_model, change, state):
        """Values of :data:`GAMMA_DIAGNOSTICS` for the accepted state ``w``."""
        c, area = self.mesh.cell_centroid, self.mesh.cell_area
        near = np.asarray(self.geom.wall_distance) < GAMMA_DIAGNOSTICS_DISTANCE
        g = np.where(near, np.asarray(g_model), 0.0)
        change = np.where(near, change, 0.0)
        i = int(np.argmax(change))
        r_nu = np.where(
            near,
            np.abs(
                self.residual(
                    w, state["beta2"], psi=state["psi"], gamma=state["gamma"]
                )[:, rans.NT]
            ),
            0.0,
        )
        j = int(np.argmax(r_nu))
        return (
            int(np.count_nonzero(g > 0.01)),
            float(np.sum(g * area)),
            int(np.count_nonzero(change > 0.05)),
            float(change[i]),
            float(c[i, 0]),
            float(c[i, 1]),
            float(r_nu[j]),
            float(c[j, 0]),
            float(c[j, 1]),
        )

    def mass(self):
        """Diagonal of the mass matrix: rho * area on u, v; area on nu_tilde."""
        area = np.asarray(self.geom.area)
        rho = self.model.density
        diag = [0.0, rho, rho] + ([1.0] if self.turbulent else [])
        return area[:, None] * np.array(diag)

    def solve_unsteady(self, w0, settings, callback=None, inner_callback=None):
        """Dual time stepping from ``w0`` to ``settings.end_time`` (D9, D37).

        Every time step solves the steady residual plus the BDF time term
        with the inner pseudo-time/Newton-Krylov iterations of
        ``settings.inner``; beta^2 follows the current state, the limiter is
        not frozen.
        """
        state = {"beta2": self.beta2(w0)}

        def residual(w):
            state["beta2"] = self.beta2(w)
            return self.residual(w, state["beta2"])

        return solve_unsteady(
            residual,
            lambda w: self.jacobian(w, state["beta2"]),
            lambda w, cfl: self.time_term(w, state["beta2"], cfl),
            self.mass(),
            w0,
            settings,
            callback=callback,
            inner_callback=inner_callback,
            jvp=lambda w, dw: self.jvp(w, dw, state["beta2"]),
        )

    def solve_explicit(
        self, w0=None, cfl=0.8, max_iterations=20000, residual_drop=8.0, callback=None
    ):
        """Explicit Runge-Kutta reference mode (D25)."""
        w0 = self.initial_state() if w0 is None else w0
        state = {"beta2": self.beta2(w0)}

        def residual(w):
            state["beta2"] = self.beta2(w)
            return self.residual(w, state["beta2"])

        return solve_explicit(
            residual,
            lambda w, r, cfl: self.explicit_update(w, r, state["beta2"], cfl),
            w0,
            cfl=cfl,
            max_iterations=max_iterations,
            residual_drop=residual_drop,
            callback=callback,
        )
