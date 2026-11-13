"""Field registry of the VTU output (D59).

Every cell field that ``output.fields`` can select is one :class:`Field`:
its name, unit, description, the model combinations it exists for and the
function that computes it. The registry drives the validation of
``output.fields``, the default selection, the writer and the field table of
docs/configuration.md (:func:`catalogue_table`), so the code and the
documentation cannot drift apart.
"""

import difflib
from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class Capabilities(NamedTuple):
    """What a case provides: decides which fields exist."""

    model: str  # "euler", "laminar" or "rans"
    transition: str  # "none", "bcm", "aft" or "lm" (only meaningful with rans)
    steady: bool
    implicit: bool  # steady with the implicit solver
    walls: bool  # output.force_boundaries is not empty
    turbulence: str = "sa-neg"  # "sa-neg" or "sst" (only meaningful with rans)
    flow: str = "incompressible"  # or "compressible" (L1)
    energy: bool = False  # incompressible energy equation (variable density)

    @classmethod
    def from_config(cls, data):
        """Capabilities of a resolved configuration (``Resolved.data``)."""
        physics = data["physics"]
        rans = physics["model"] == "rans"
        steady = data["time"]["mode"] == "steady"
        return cls(
            physics["model"],
            physics["transition"]["model"] if rans else "none",
            steady,
            steady and data["solver"]["method"] == "implicit",
            bool(data["output"]["force_boundaries"]),
            physics["turbulence"] if rans else "none",
            physics.get("flow", "incompressible"),
            bool(physics.get("energy", False))
            and physics.get("flow", "incompressible") == "incompressible",
        )


class Field(NamedTuple):
    """One selectable output field."""

    name: str
    unit: str
    description: str
    needs: str  # short text of the availability rule (documentation, errors)
    available: Callable  # Capabilities -> bool
    compute: Callable  # FieldContext -> ndarray, (n_cells,) or (n_cells, 2)


class Freestream(NamedTuple):
    """Freestream state of the derived fields (cp, total pressure), non-dimensional.

    ``q`` is the reference dynamic pressure 0.5 rho_ref U_ref^2 of the force
    coefficients and of ``cp_wall`` (0.5 in the non-dimensional variables).
    """

    pressure: float
    density: float
    speed: float
    temperature: float = 1.0
    q: float = 0.5
    gamma: float = 1.4  # compressible only
    compressible: bool = False

    @property
    def mach(self):
        """Freestream Mach number (compressible), from p = rho R T."""
        return self.speed / np.sqrt(self.gamma * self.pressure / self.density)

    @property
    def total_pressure(self):
        """Freestream total pressure (incompressible: p + rho U^2 / 2)."""
        return _total_pressure(
            self.pressure, self.density, self.speed, self.gamma, self.compressible
        )

    @classmethod
    def from_config(cls, data, scaling, model, compressible):
        """Build the freestream of the single farfield stream, else the reference.

        Incompressible: farfield pressure (gauge) and velocity, the fluid
        density; without exactly one farfield stream p = 0 and U = U_ref (as
        ``cp_wall``). Compressible: farfield p, T and Mach, rho = p / (R T);
        without exactly one farfield stream the reference state.
        """
        streams = {
            tuple(b.get(k) for k in ("pressure", "velocity", "mach", "temperature"))
            for b in data["boundary"].values()
            if b["type"] == "farfield"
        }
        one = streams.pop() if len(streams) == 1 else None
        if compressible:
            r, g = model.gas_constant, model.gamma
            if one is None:
                return cls(r, 1.0, 1.0, 1.0, 0.5, g, True)
            p = one[0] / scaling.pressure
            t = one[3] / scaling.temperature
            return cls(p, p / (r * t), one[2] * np.sqrt(g * r * t), t, 0.5, g, True)
        if one is None:
            return cls(0.0, model.density, 1.0)
        return cls(one[0] / scaling.pressure, model.density, one[1] / scaling.velocity)


def _total_pressure(p, rho, speed, gamma, compressible):
    if compressible:
        mach2 = speed**2 * rho / (gamma * p)
        return p * (1.0 + 0.5 * (gamma - 1.0) * mach2) ** (gamma / (gamma - 1.0))
    return p + 0.5 * rho * speed**2


class FieldContext:
    """State a field is computed from; wall quantities and gradients once."""

    def __init__(
        self, problem, w, scaling, monitor=None, cell_residual=None, freestream=None
    ):
        self.problem, self.w, self.scaling = problem, np.asarray(w), scaling
        self.monitor = monitor
        self.cell_residual = cell_residual
        self.freestream = freestream
        self._wall = None
        self._gradient = None

    def gradient(self):
        """Least-squares cell gradients (n_cells, n_var, 2), non-dimensional."""
        if self._gradient is None:
            self._gradient = self.problem.gradient(self.w)
        return self._gradient

    def velocity_gradient(self):
        """(du/dx, du/dy, dv/dx, dv/dy) per cell, non-dimensional."""
        g = self.gradient()
        return g[:, 1, 0], g[:, 1, 1], g[:, 2, 0], g[:, 2, 1]

    def density_nd(self):
        """Density per cell, non-dimensional."""
        m = self.problem.model
        if self.problem.compressible:
            return self.w[:, 0] / (m.gas_constant * self.w[:, 3])
        if getattr(self.problem, "thermal", False):
            return _vd().cell_properties(m, self.w)["density"]
        return np.full(self.w.shape[0], m.density)

    def total_pressure_nd(self):
        """Total pressure per cell, non-dimensional (as :class:`Freestream`)."""
        speed = np.hypot(self.w[:, 1], self.w[:, 2])
        gamma = getattr(self.problem.model, "gamma", 1.4)
        return _total_pressure(
            self.w[:, 0], self.density_nd(), speed, gamma, self.problem.compressible
        )

    @property
    def nu_si(self):
        """Laminar kinematic viscosity in m^2/s."""
        m = self.problem.model
        return m.viscosity / m.density * self.scaling.viscosity / self.scaling.density

    def wall(self):
        """Per-cell wall quantities (non-dimensional), 0 away from walls.

        A cell with several wall faces gets the length-weighted mean.
        """
        if self._wall is None:
            self._wall = _wall_cells(self)
        return self._wall


def _wall_cells(ctx):
    problem, monitor = ctx.problem, ctx.monitor
    geom = problem.geom
    faces = problem.boundary_faces(monitor.boundaries)
    d = _fz().wall_distribution(
        problem.wall_loads(ctx.w), geom, faces, monitor.q, monitor.p_ref
    )
    normal = np.asarray(geom.b_normal)[faces]
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    a = np.radians(monitor.alpha)
    # Attached flow positive: orient the tangent along the freestream.
    sign = np.where(tangent @ np.array([np.cos(a), np.sin(a)]) >= 0.0, 1.0, -1.0)
    values = {
        "cf": d["cf"] * sign,
        "cp_wall": d["cp"],
        "tau_wall_x": d["tau_x"],
        "tau_wall_y": d["tau_y"],
    }
    owner = np.asarray(geom.b_owner)[faces]
    length = np.asarray(geom.b_length)[faces]
    if problem.turbulent:
        tau = np.hypot(d["tau_x"], d["tau_y"])
        values["y_plus"] = problem.wall_y_plus(ctx.w, faces, tau)
    weight = np.zeros(problem.mesh.n_cells)
    np.add.at(weight, owner, length)
    out = {}
    for key, face_values in values.items():
        cell = np.zeros(problem.mesh.n_cells)
        np.add.at(cell, owner, face_values * length)
        out[key] = np.divide(cell, weight, out=np.zeros_like(cell), where=weight > 0)
    return out


def _fz():
    from pyaeroflow.physics import forces

    return forces


def _vd():
    from pyaeroflow.physics import variable_density

    return variable_density


# -- compute functions --------------------------------------------------------


def _pressure(c):
    return c.w[:, 0] * c.scaling.pressure


def _velocity(c):
    return c.w[:, 1:3] * c.scaling.velocity


def _speed(c):
    return np.hypot(c.w[:, 1], c.w[:, 2]) * c.scaling.velocity


def _density(c):
    return c.density_nd() * c.scaling.density


def _temperature(c):
    return c.w[:, 3] * c.scaling.temperature


def _mach(c):
    m = c.problem.model
    speed = np.hypot(c.w[:, 1], c.w[:, 2])
    return speed / np.sqrt(m.gamma * m.gas_constant * c.w[:, 3])


def _vorticity(c):
    return (
        c.problem.vorticity(c.w, c.gradient()) * c.scaling.velocity / c.scaling.length
    )


def _q_criterion(c):
    # Q = (|Omega|^2 - |S|^2) / 2 with the full strain (dilatation included),
    # Frobenius norms; in 2D Omega_12 = (du/dy - dv/dx) / 2.
    ux, uy, vx, vy = c.velocity_gradient()
    omega2 = 0.5 * (uy - vx) ** 2
    strain2 = ux**2 + vy**2 + 0.5 * (uy + vx) ** 2
    return 0.5 * (omega2 - strain2) * (c.scaling.velocity / c.scaling.length) ** 2


def _strain_rate(c):
    ux, uy, vx, vy = c.velocity_gradient()
    s = np.sqrt(2.0 * (ux**2 + vy**2) + (uy + vx) ** 2)
    return s * c.scaling.velocity / c.scaling.length


def _divergence(c):
    ux, _, _, vy = c.velocity_gradient()
    return (ux + vy) * c.scaling.velocity / c.scaling.length


def _cp(c):
    f = c.freestream
    return (c.w[:, 0] - f.pressure) / f.q


def _total_pressure_field(c):
    return c.total_pressure_nd() * c.scaling.pressure


def _total_pressure_loss(c):
    f = c.freestream
    return (f.total_pressure - c.total_pressure_nd()) / f.q


def _entropy(c):
    # (s - s_inf) / c_v = ln[(p / p_inf) (rho_inf / rho)^gamma]
    f, g = c.freestream, c.problem.model.gamma
    return np.log(c.w[:, 0] / f.pressure * (f.density / c.density_nd()) ** g)


def _density_gradient(c):
    # rho = p / (R T): grad rho = grad p / (R T) - p grad T / (R T^2)
    r = c.problem.model.gas_constant
    g = c.gradient()
    p, t = c.w[:, 0], c.w[:, 3]
    grad = g[:, 0, :] / (r * t)[:, None] - (p / (r * t**2))[:, None] * g[:, 3, :]
    return np.hypot(grad[:, 0], grad[:, 1]) * c.scaling.density / c.scaling.length


def _pressure_gradient_nd(c):
    return c.gradient()[:, 0, :]


def _pressure_gradient(c):
    return _pressure_gradient_nd(c) * c.scaling.pressure / c.scaling.length


def _pressure_gradient_magnitude(c):
    g = _pressure_gradient_nd(c)
    return np.hypot(g[:, 0], g[:, 1]) * c.scaling.pressure / c.scaling.length


def _kinematic_viscosity(c):
    if getattr(c.problem, "thermal", False):  # mu(T) / rho(T) per cell
        nu = c.problem.kinematic_viscosity(c.w)
        return nu * c.scaling.viscosity / c.scaling.density
    if c.problem.compressible:  # mu(T) / rho per cell
        from pyaeroflow.physics import compressible as comp

        m = c.problem.model
        mu = np.asarray(comp.laminar_viscosity(m, c.w[:, 3]))
        rho = c.w[:, 0] / (m.gas_constant * c.w[:, 3])
        return mu / rho * c.scaling.viscosity / c.scaling.density
    return np.full(c.w.shape[0], c.nu_si)


def _nu_tilde(c):
    col = c.problem.turbulence_column
    return c.w[:, col] * c.scaling.viscosity / c.scaling.density


def _eddy_viscosity(c):
    return c.problem.eddy_viscosity(c.w) * c.scaling.viscosity / c.scaling.density


def _eddy_viscosity_ratio(c):
    return c.problem.eddy_viscosity(c.w) / c.problem.kinematic_viscosity(c.w)


def _effective_viscosity(c):
    nu_t = _eddy_viscosity(c) if c.problem.turbulent else 0.0
    return _kinematic_viscosity(c) + nu_t


def _dynamic_eddy_viscosity(c):
    nu_t = c.problem.eddy_viscosity(c.w) * c.scaling.viscosity / c.scaling.density
    if c.problem.compressible:
        m = c.problem.model
        rho = c.w[:, 0] / (m.gas_constant * c.w[:, 3])
    else:
        rho = c.problem.model.density
    return rho * c.scaling.density * nu_t


def _turbulent_kinetic_energy(c):
    return c.w[:, c.problem.turbulence_column] * c.scaling.velocity**2


def _specific_dissipation(c):
    col = c.problem.turbulence_column + 1
    return c.w[:, col] * c.scaling.velocity / c.scaling.length


def _blending_f1(c):
    return c.problem.blending_f1(c.w)


def _wall_distance(c):
    return np.asarray(c.problem.geom.wall_distance) * c.scaling.length


def _intermittency(c):
    return c.problem.intermittency(c.w)


def _amplification_factor(c):
    return c.w[:, 5]


def _re_theta_t(c):
    return c.w[:, 6]


def _effective_intermittency(c):
    return c.problem.effective_intermittency(c.w)


def _residual(column, scale=None, turbulence=False):
    def compute(c):
        # Residual per unit cell volume (area in 2D), non-dimensional; ``scale``
        # undoes a row scaling of the residual. ``turbulence``: ``column``
        # counts from the first turbulence column (3 incompressible, 4
        # compressible).
        col = column + (c.problem.turbulence_column if turbulence else 0)
        r = np.asarray(c.cell_residual())[:, col] / np.asarray(c.problem.geom.area)
        return r if scale is None else r / scale(c)

    return compute


def _nu_nd(c):
    # The SST omega residual is solved multiplied by nu (rans_sst).
    return c.problem.model.viscosity / c.problem.model.density


def _ret_scale(c):
    # The Langtry-Menter Re_thetat residual is solved multiplied by RET_SCALE.
    return 1.0e-3


def _wall(key):
    return lambda c: c.wall()[key]


# -- availability rules -------------------------------------------------------


def _always(caps):
    return True


def _viscous(caps):
    return caps.model != "euler"


def _rans(caps):
    return caps.model == "rans"


def _sa(caps):
    return caps.model == "rans" and caps.turbulence != "sst"


def _sst(caps):
    return caps.model == "rans" and caps.turbulence == "sst"


def _implicit_sa(caps):
    return caps.implicit and _sa(caps)


def _implicit_sst(caps):
    return caps.implicit and _sst(caps)


def _transition(caps):
    return caps.model == "rans" and caps.transition != "none"


def _aft(caps):
    return caps.model == "rans" and caps.transition == "aft"


def _implicit(caps):
    return caps.implicit


def _compressible(caps):
    return caps.flow == "compressible"


def _thermal(caps):
    """Compressible flow or the incompressible energy equation (a T column)."""
    return caps.flow == "compressible" or caps.energy


def _implicit_incompressible(caps):
    return caps.implicit and caps.flow != "compressible"


def _implicit_compressible(caps):
    return caps.implicit and caps.flow == "compressible"


def _implicit_aft(caps):
    return caps.implicit and _aft(caps)


def _lm(caps):
    return caps.model == "rans" and caps.transition == "lm"


def _implicit_lm(caps):
    return caps.implicit and _lm(caps)


def _wall_any(caps):
    return caps.walls


def _wall_viscous(caps):
    return caps.walls and caps.model != "euler"


def _wall_rans(caps):
    return caps.walls and caps.model == "rans"


FIELDS = (
    Field("pressure", "Pa", "Static pressure", "always", _always, _pressure),
    Field("velocity", "m/s", "Velocity vector", "always", _always, _velocity),
    Field("velocity_magnitude", "m/s", "Speed", "always", _always, _speed),
    Field(
        "density",
        "kg/m^3",
        "Density p / (R T) (incompressible energy equation: rho(T) of "
        "fluid.density_model)",
        "compressible or physics.energy",
        _thermal,
        _density,
    ),
    Field(
        "temperature",
        "K",
        "Static temperature",
        "compressible or physics.energy",
        _thermal,
        _temperature,
    ),
    Field(
        "mach",
        "1",
        "Mach number |v| / c",
        "compressible",
        _compressible,
        _mach,
    ),
    Field(
        "cp",
        "1",
        "Pressure coefficient (p - p_inf) / q_inf, freestream of the farfield",
        "always",
        _always,
        _cp,
    ),
    Field(
        "total_pressure",
        "Pa",
        "p + rho |v|^2 / 2 (compressible: isentropic p (1 + (gamma-1)/2 M^2)^"
        "(gamma/(gamma-1)))",
        "always",
        _always,
        _total_pressure_field,
    ),
    Field(
        "total_pressure_loss",
        "1",
        "(p0_inf - p0) / q_inf: wakes, separation, numerical losses",
        "always",
        _always,
        _total_pressure_loss,
    ),
    Field(
        "entropy",
        "1",
        "(s - s_inf) / c_v = ln[(p / p_inf) (rho_inf / rho)^gamma]",
        "compressible",
        _compressible,
        _entropy,
    ),
    Field(
        "density_gradient",
        "kg/m^4",
        "|grad rho| (numerical schlieren) from the least-squares gradient",
        "compressible",
        _compressible,
        _density_gradient,
    ),
    Field(
        "pressure_gradient",
        "Pa/m",
        "grad p (vector) from the least-squares gradient",
        "always",
        _always,
        _pressure_gradient,
    ),
    Field(
        "pressure_gradient_magnitude",
        "Pa/m",
        "|grad p| from the least-squares gradient (refinement indicator)",
        "always",
        _always,
        _pressure_gradient_magnitude,
    ),
    Field(
        "vorticity",
        "1/s",
        "dv/dx - du/dy from the least-squares gradient (signed)",
        "always",
        _always,
        _vorticity,
    ),
    Field(
        "q_criterion",
        "1/s^2",
        "Q = (|Omega|^2 - |S|^2) / 2 from the velocity gradient; Q > 0: vortex",
        "always",
        _always,
        _q_criterion,
    ),
    Field(
        "strain_rate",
        "1/s",
        "sqrt(2 S_ij S_ij) from the velocity gradient",
        "always",
        _always,
        _strain_rate,
    ),
    Field(
        "divergence",
        "1/s",
        "du/dx + dv/dy (compressible: dilatation; incompressible: mass error)",
        "always",
        _always,
        _divergence,
    ),
    Field(
        "kinematic_viscosity",
        "m^2/s",
        "Laminar kinematic viscosity (constant; compressible: mu(T) / rho)",
        "laminar, rans",
        _viscous,
        _kinematic_viscosity,
    ),
    Field(
        "eddy_viscosity",
        "m^2/s",
        "Kinematic eddy viscosity nu_t",
        "rans",
        _rans,
        _eddy_viscosity,
    ),
    Field(
        "eddy_viscosity_ratio",
        "1",
        "nu_t / nu",
        "rans",
        _rans,
        _eddy_viscosity_ratio,
    ),
    Field(
        "effective_viscosity",
        "m^2/s",
        "nu + nu_t (the laminar value without a turbulence model)",
        "laminar, rans",
        _viscous,
        _effective_viscosity,
    ),
    Field(
        "mu_t",
        "Pa s",
        "Dynamic eddy viscosity rho nu_t",
        "rans",
        _rans,
        _dynamic_eddy_viscosity,
    ),
    Field(
        "nu_tilde",
        "m^2/s",
        "SA working variable",
        "rans with turbulence = sa-neg",
        _sa,
        _nu_tilde,
    ),
    Field(
        "k",
        "m^2/s^2",
        "Turbulent kinetic energy (SST)",
        "rans with turbulence = sst",
        _sst,
        _turbulent_kinetic_energy,
    ),
    Field(
        "omega",
        "1/s",
        "Specific dissipation rate (SST)",
        "rans with turbulence = sst",
        _sst,
        _specific_dissipation,
    ),
    Field(
        "blending_f1",
        "1",
        "Menter's blending function F1 (1: k-omega near the wall, 0: k-epsilon)",
        "rans with turbulence = sst",
        _sst,
        _blending_f1,
    ),
    Field(
        "wall_distance",
        "m",
        "Distance to the nearest wall used by the turbulence model",
        "rans",
        _rans,
        _wall_distance,
    ),
    Field(
        "intermittency",
        "1",
        "Transition intermittency gamma",
        "rans with a transition model",
        _transition,
        _intermittency,
    ),
    Field(
        "amplification_factor",
        "1",
        "Amplification factor n of SA-AFT",
        "rans with transition.model = aft",
        _aft,
        _amplification_factor,
    ),
    Field(
        "re_theta_t",
        "1",
        "Transported transition-onset momentum-thickness Reynolds number "
        "Re_thetat of Langtry-Menter",
        "rans with transition.model = lm",
        _lm,
        _re_theta_t,
    ),
    Field(
        "intermittency_effective",
        "1",
        "Effective intermittency max(gamma, gamma_sep) on the SST production "
        "(gamma_sep > 1 marks separation-induced transition)",
        "rans with transition.model = lm",
        _lm,
        _effective_intermittency,
    ),
    Field(
        "residual_p",
        "1",
        "Pressure (continuity) residual per cell volume",
        "steady, implicit, incompressible",
        _implicit_incompressible,
        _residual(0),
    ),
    Field(
        "residual_u",
        "1",
        "x-momentum residual per cell volume",
        "steady, implicit, incompressible",
        _implicit_incompressible,
        _residual(1),
    ),
    Field(
        "residual_v",
        "1",
        "y-momentum residual per cell volume",
        "steady, implicit, incompressible",
        _implicit_incompressible,
        _residual(2),
    ),
    Field(
        "residual_rho",
        "1",
        "Continuity residual per cell volume",
        "steady, implicit, compressible",
        _implicit_compressible,
        _residual(0),
    ),
    Field(
        "residual_rhou",
        "1",
        "x-momentum residual per cell volume",
        "steady, implicit, compressible",
        _implicit_compressible,
        _residual(1),
    ),
    Field(
        "residual_rhov",
        "1",
        "y-momentum residual per cell volume",
        "steady, implicit, compressible",
        _implicit_compressible,
        _residual(2),
    ),
    Field(
        "residual_rhoE",
        "1",
        "Energy residual per cell volume",
        "steady, implicit, compressible",
        _implicit_compressible,
        _residual(3),
    ),
    Field(
        "residual_nu_tilde",
        "1",
        "nu_tilde residual per cell volume",
        "steady, implicit, turbulence = sa-neg",
        _implicit_sa,
        _residual(0, turbulence=True),
    ),
    Field(
        "residual_k",
        "1",
        "k residual per cell volume",
        "steady, implicit, turbulence = sst",
        _implicit_sst,
        _residual(0, turbulence=True),
    ),
    Field(
        "residual_omega",
        "1",
        "omega residual per cell volume",
        "steady, implicit, turbulence = sst",
        _implicit_sst,
        _residual(1, _nu_nd, turbulence=True),
    ),
    Field(
        "residual_gamma",
        "1",
        "log-intermittency residual per cell volume",
        "steady, implicit, transition.model = aft",
        _implicit_aft,
        _residual(4),
    ),
    Field(
        "residual_n",
        "1",
        "Amplification-factor residual per cell volume",
        "steady, implicit, transition.model = aft",
        _implicit_aft,
        _residual(5),
    ),
    Field(
        "residual_intermittency",
        "1",
        "Intermittency residual per cell volume",
        "steady, implicit, transition.model = lm",
        _implicit_lm,
        _residual(5),
    ),
    Field(
        "residual_re_theta_t",
        "1",
        "Re_thetat residual per cell volume",
        "steady, implicit, transition.model = lm",
        _implicit_lm,
        _residual(6, _ret_scale),
    ),
    Field(
        "cf",
        "1",
        "Skin-friction coefficient with attached flow positive on every side "
        "(the wall CSV keeps the sign of the wall traversal); wall cells only",
        "wall in output.force_boundaries, laminar or rans",
        _wall_viscous,
        _wall("cf"),
    ),
    Field(
        "cp_wall",
        "1",
        "Pressure coefficient at the wall; wall cells only",
        "wall in output.force_boundaries",
        _wall_any,
        _wall("cp_wall"),
    ),
    Field(
        "tau_wall_x",
        "1",
        "Wall shear stress x component (non-dimensional, rho U^2); wall cells only",
        "wall in output.force_boundaries, laminar or rans",
        _wall_viscous,
        _wall("tau_wall_x"),
    ),
    Field(
        "tau_wall_y",
        "1",
        "Wall shear stress y component (non-dimensional, rho U^2); wall cells only",
        "wall in output.force_boundaries, laminar or rans",
        _wall_viscous,
        _wall("tau_wall_y"),
    ),
    Field(
        "y_plus",
        "1",
        "y+ of the first cell centre; wall cells only",
        "wall in output.force_boundaries, rans",
        _wall_rans,
        _wall("y_plus"),
    ),
)

BY_NAME = {f.name: f for f in FIELDS}

# Fields with two components (written as 3-component VTU vectors).
VECTOR_NAMES = ("velocity", "pressure_gradient")

# The fields written without ``output.fields`` (what the VTU always held).
DEFAULT_NAMES = (
    "pressure",
    "velocity",
    "velocity_magnitude",
    "density",
    "mach",
    "nu_tilde",
    "k",
    "omega",
    "eddy_viscosity_ratio",
    "intermittency",
    "amplification_factor",
    "re_theta_t",
    "intermittency_effective",
)


def available(caps):
    """Names of the fields that exist for ``caps``, in registry order."""
    return [f.name for f in FIELDS if f.available(caps)]


def default_fields(caps):
    """Default selection: pressure, velocity, speed and the turbulence fields.

    The incompressible energy equation adds the temperature.
    """
    names = [n for n in DEFAULT_NAMES if BY_NAME[n].available(caps)]
    if caps.energy:
        names.insert(names.index("density") + 1, "temperature")
    return names


def resolve_fields(requested, caps):
    """Return the selected field names in registry order and the problems found.

    ``requested`` is the ``output.fields`` array; ``None`` selects the
    default set and ``["all"]`` every available field. An unknown name gets a
    "did you mean" hint, a known field that the case does not provide names
    what it needs.
    """
    problems = []
    if requested is None:
        return default_fields(caps), problems
    if "all" in requested:
        return available(caps), problems
    chosen = set()
    for name in requested:
        field = BY_NAME.get(name)
        if field is None:
            close = difflib.get_close_matches(str(name), list(BY_NAME), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            problems.append(f"output.fields: unknown field '{name}'{hint}")
        elif not field.available(caps):
            problems.append(
                f"output.fields: '{name}' needs {field.needs} "
                f"(this case: model {caps.model}"
                + (", compressible" if caps.flow == "compressible" else "")
                + (
                    f", turbulence {caps.turbulence}, transition {caps.transition}"
                    if caps.model == "rans"
                    else ""
                )
                + f", {'steady' if caps.steady else 'unsteady'}"
                + (", implicit" if caps.implicit else "")
                + ")"
            )
        else:
            chosen.add(name)
    return [f.name for f in FIELDS if f.name in chosen], problems


def compute_fields(names, ctx):
    """``{name: array}`` of the named fields for the write."""
    return {n: np.asarray(BY_NAME[n].compute(ctx)) for n in names}


def n_columns(names):
    """(scalar fields, vector fields) among ``names``."""
    vectors = sum(1 for n in names if n in VECTOR_NAMES)
    return len(names) - vectors, vectors


def catalogue_table():
    """Markdown table of all fields for docs/configuration.md."""
    cases = (
        ("euler", Capabilities("euler", "none", True, True, True)),
        (
            "compressible euler",
            Capabilities("euler", "none", True, True, True, "none", "compressible"),
        ),
        ("laminar", Capabilities("laminar", "none", True, True, True)),
        (
            "compressible laminar",
            Capabilities("laminar", "none", True, True, True, "none", "compressible"),
        ),
        (
            "laminar + energy",
            Capabilities(
                "laminar", "none", True, True, True, "none", "incompressible", True
            ),
        ),
        ("SA", Capabilities("rans", "none", True, True, True)),
        (
            "compressible SA",
            Capabilities("rans", "none", True, True, True, "sa-neg", "compressible"),
        ),
        ("SA + BCM", Capabilities("rans", "bcm", True, True, True)),
        ("SA + AFT", Capabilities("rans", "aft", True, True, True)),
        ("SST", Capabilities("rans", "none", True, True, True, "sst")),
        (
            "compressible SST",
            Capabilities("rans", "none", True, True, True, "sst", "compressible"),
        ),
        ("SST + LM", Capabilities("rans", "lm", True, True, True, "sst")),
    )
    head = "| Field | Unit | Description | Needs | " + " | ".join(c[0] for c in cases)
    lines = [head + " |", "|" + "---|" * (4 + len(cases))]
    for f in FIELDS:
        marks = " | ".join("x" if f.available(c[1]) else "" for c in cases)
        lines.append(
            f"| `{f.name}` | {f.unit} | {f.description} | {f.needs} | {marks} |"
        )
    return "\n".join(lines)


def case_listing(data):
    """Text listing of the fields a resolved case provides (``*`` = selected).

    ``data`` is ``Resolved.data``. Fields of the registry that the case does
    not provide are listed after them with what they need.
    """
    caps = Capabilities.from_config(data)
    chosen = set(data["output"]["fields"])
    here = [f for f in FIELDS if f.available(caps)]
    rest = [f for f in FIELDS if not f.available(caps)]
    width = max(len(f.name) for f in FIELDS)
    lines = ["Fields this case provides (* = in output.fields):"]
    lines += [
        f"  {'*' if f.name in chosen else ' '} {f.name:<{width}}  {f.unit:<8} "
        f"{f.description}"
        for f in here
    ]
    lines += ["", "Fields it does not provide (what they need):"]
    lines += [f"    {f.name:<{width}}  {f.needs}" for f in rest]
    return "\n".join(lines)
