"""Case file for an airfoil mesh (``pyaeroflow init-case``).

The settings per model are the ones of the validated example cases
(``examples/cases``): the numerics and solver blocks of the PyAero cases and
the freestream turbulence each model needs (``docs/pyaero.md``).
"""

import os
from pathlib import Path

from pyaeroflow.mesh import read_mesh
from pyaeroflow.meshcheck import PYAERO_WALL, check_mesh

PYAERO_FAR = "inlet"

MODELS = {
    "euler": "inviscid flow, slip wall",
    "laminar": "laminar Navier-Stokes",
    "sa": "Spalart-Allmaras (SA-neg), fully turbulent",
    "bcm": "SA with the algebraic BCM transition model",
    "aft": "SA-AFT2019b transition (e^N-like, N_crit from Tu)",
    "sst": "k-omega SST (2003m), fully turbulent",
    "lm": "SST with Langtry-Menter gamma-Re_theta transition",
}

_FREESTREAM_TURBULENCE = {
    "sa": "{ nu_tilde_ratio = 3.0 }",
    "bcm": "{ nu_tilde_ratio = 0.1 }   # laminar freestream for the transition model",
    "aft": "{ nu_tilde_ratio = 0.1 }   # laminar freestream for the transition model",
}


def make_case(
    mesh_file,
    case_file,
    model,
    reynolds=None,
    alpha=0.0,
    chord=None,
    wall=None,
    turbulence_intensity=0.001,
    scale=1.0,
    name=None,
):
    """Return the text of a case file for a PyAero (or Gmsh) airfoil mesh.

    Parameters
    ----------
    mesh_file : str or Path
        ``.su2`` or ``.msh`` file.
    case_file : str or Path
        Where the case will be written; the mesh path in it is relative to
        this location.
    model : str
        One of :data:`MODELS`.
    reynolds : float, optional
        Chord Reynolds number; required unless ``model`` is ``"euler"``.
    alpha : float
        Angle of attack in degrees (a polar overrides it).
    chord : float, optional
        Chord in metres; default the x extent of the wall after ``scale``.
    wall : str, optional
        Name of the airfoil boundary; default ``"airfoil"``.
    turbulence_intensity : float
        Freestream Tu as a fraction, for the transition models.
    scale : float
        Mesh scale factor (``mesh.scale``).
    name : str, optional
        Case name; default ``<mesh stem>_<model>``.

    Returns
    -------
    text : str
    notes : list of str
        Things the user should know (findings of the mesh check, defaults).
    """
    if model not in MODELS:
        raise ValueError(f"unknown model '{model}' (choose from {', '.join(MODELS)})")
    if model == "euler":
        reynolds = None  # inviscid: no Reynolds number, no y+ estimate
    elif not reynolds:
        raise ValueError(f"--re is required for model '{model}'")
    mesh_file, case_file = Path(mesh_file), Path(case_file)
    mesh = read_mesh(mesh_file, scale)
    wall = wall or (PYAERO_WALL if PYAERO_WALL in mesh.boundary_faces else None)
    if wall is None or wall not in mesh.boundary_faces:
        raise ValueError(
            f"no airfoil boundary '{wall or PYAERO_WALL}' in the mesh; give the "
            f"wall with --wall (boundaries: {', '.join(mesh.boundary_faces)})"
        )
    far = [b for b in mesh.boundary_faces if b != wall]
    if not far:
        raise ValueError("the mesh has no boundary besides the wall")

    report = check_mesh(mesh, wall=wall, reynolds=reynolds, chord=chord)
    chord = report.chord
    nodes = mesh.nodes[mesh.face_nodes[mesh.boundary_faces[wall]].ravel()]
    x_le = float(nodes[:, 0].min())
    notes = [
        f"{level}: {text}"
        for level, text in report.findings
        if model != "euler" or "--re" not in text
    ]
    notes.insert(0, f"chord {chord:.6g} m taken from the mesh, moment center at 25 %")
    name = name or f"{mesh_file.stem}_{model}"
    rel = os.path.relpath(mesh_file.resolve(), case_file.resolve().parent)
    rho, velocity = 1.0, 1.0
    viscosity = rho * velocity * chord / reynolds if reynolds else None
    return _text(
        model=model,
        name=name,
        mesh=Path(rel).as_posix(),
        scale=scale,
        wall=wall,
        far=far,
        initial=PYAERO_FAR if PYAERO_FAR in far else far[0],
        chord=chord,
        x_quarter=x_le + 0.25 * chord,
        viscosity=viscosity,
        reynolds=reynolds,
        alpha=alpha,
        tu=turbulence_intensity,
        rho=rho,
        velocity=velocity,
    ), notes


def _text(**k):
    model, rans = k["model"], k["model"] in ("sa", "bcm", "aft", "sst", "lm")
    out = [
        f"# {k['name']}: {MODELS[model]}.",
        "# Written by 'pyaeroflow init-case'; every key is explained in",
        "# docs/configuration.md. Run it with",
        f"#   pyaeroflow run {k['name']}.toml",
        f"#   pyaeroflow polar {k['name']}.toml --alpha 0 2 4 6 8",
        "",
        "[case]",
        f'name = "{k["name"]}"',
        "",
        "[mesh]",
        f'file = "{k["mesh"]}"',
    ]
    if k["scale"] != 1.0:
        out.append(f"scale = {k['scale']:g}")
    out += ["", "[physics]"]
    out.append(
        {
            "euler": 'model = "euler"',
            "laminar": 'model = "laminar"',
        }.get(model, 'model = "rans"')
    )
    if rans:
        turb = "sst" if model in ("sst", "lm") else "sa-neg"
        out.append(f'turbulence = "{turb}"')
    if model in ("bcm", "aft", "lm"):
        out += ["", "[physics.transition]", f'model = "{model}"']
        if model != "lm":  # the LM model takes Tu from the local k
            out.append(
                f"turbulence_intensity = {k['tu']:g}      # fraction: "
                f"{100 * k['tu']:g} %"
            )
    out += ["", "[fluid]", f"density = {k['rho']:g}                     # kg/m^3"]
    if k["viscosity"] is not None:
        out.append(
            f"viscosity = {k['viscosity']:.8g}   # Pa s: Re = {k['reynolds']:g} "
            f"with U = {k['velocity']:g} m/s, c = {k['chord']:.6g} m"
        )
    out += [
        "",
        "[reference]",
        f"velocity = {k['velocity']:g}                    # m/s",
        f"length = {k['chord']:.6g}                     # m, chord",
        f"moment_center = [{k['x_quarter']:.6g}, 0.0]    # m, quarter chord "
        "(also the vortex location)",
        "vortex_correction = true         # farfield: freestream + lift-induced flow",
        "",
        "[initial]",
        f'from_boundary = "{k["initial"]}"',
        "",
        f"[boundary.{_key(k['wall'])}]",
        f'type = "{"slip-wall" if model == "euler" else "wall"}"',
    ]
    freestream = _FREESTREAM_TURBULENCE.get(
        model,
        f"{{ turbulent_viscosity_ratio = 1.0, intensity = {k['tu']:g} }}",
    )
    for b in k["far"]:
        out += [
            "",
            f"[boundary.{_key(b)}]",
            'type = "farfield"',
            f"velocity = {k['velocity']:g}                    # m/s",
            f"angle_of_attack = {k['alpha']:g}             # deg, a polar overrides it",
            "pressure = 0.0                   # Pa",
        ]
        if rans:
            out.append(f"turbulence = {freestream}")
    out += [""]
    if model == "euler":
        out += [
            "[numerics]",
            "limiter_freeze = { after_residual_drop = 2.0 }",
            "",
            "[solver]",
            "residual_drop = 6.0",
        ]
    elif model == "laminar":
        out += [
            "[solver]",
            "residual_drop = 7.0",
            "max_iterations = 2000",
            'cfl = { control = "ser" }',
        ]
    else:
        out += [
            "[numerics]",
            'limiter = "venkatakrishnan"       # local threshold for the start-up',
            "limiter_freeze = { after_residual_drop = 1.5, after_iterations = 40, "
            "release = true }",
            "",
            "[solver]",
            "residual_drop = 7.0",
            "max_iterations = 2000",
            'cfl = { control = "growth", max = 1.0e5 }   # no undamped Newton steps'
            if model == "lm"
            else 'cfl = { control = "ser" }',
            "newton_krylov = { switch_at_residual_drop = 1.5, "
            "switch_at_iteration = 40 }",
            '# linear_solver = { type = "pardiso" }   # needs the pypardiso package',
        ]
    out += [
        "",
        "[output]",
        f'directory = "output/{k["name"]}"',
        "restart_interval = 50",
        f'force_boundaries = ["{k["wall"]}"]    '
        f"# wall CSV: {k['name']}.wall_{k['wall']}.csv",
        "",
    ]
    return "\n".join(out)


def _key(name):
    """TOML key for a boundary name (quoted when it has special characters)."""
    ok = name.replace("_", "").replace("-", "").isalnum()
    return name if ok else f'"{name}"'
