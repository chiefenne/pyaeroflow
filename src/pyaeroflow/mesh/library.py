"""Meshes of the example and template cases, made without external files.

Every entry of :data:`LIBRARY` builds a mesh from parameters only (no
coordinate file, no download), so a case that uses one runs after a plain
``pip install``. ``build(name, **parameters)`` returns the
:class:`~pyaeroflow.mesh.Mesh`; ``pyaeroflow mesh-library`` lists the names.
The scripts under ``examples/`` call the same functions.
"""

import numpy as np

from pyaeroflow.mesh.airfoil import (
    _check_folds,
    _loop_normals,
    _smooth,
    airfoil_ogrid,
    airfoil_points,
    naca0012_ogrid,
)
from pyaeroflow.mesh.cylinder import annulus, split_boundary
from pyaeroflow.mesh.generators import geometric_growth, rectangle, structured
from pyaeroflow.mesh.mesh import Mesh

__all__ = [
    "LIBRARY",
    "build",
    "jet_crossflow_mesh",
    "heated_plate_mesh",
    "naca0012_shedding_mesh",
    "plate_lm_mesh",
]


def _graded(length, first, n):
    """Geometrically growing widths of n cells from ``first``, summing to ``length``."""
    r = geometric_growth(length, first, n)
    widths = first * r ** np.arange(n)
    return widths * length / widths.sum()


def jet_crossflow_mesh(
    x_up=10.0, x_down=80.0, height=20.0, n_slot=32, dy1=0.01, ny=110
):
    """Jet in crossflow over a flat plate: quad mesh with a slot marker.

    Length unit: the slot width d = 1. The slot spans x = -0.5 .. 0.5 on the
    bottom wall (y = 0). The domain runs from x = -``x_up`` (inflow) to
    ``x_down`` (outflow) and up to ``height``. Cells are uniform across the
    slot (``n_slot`` faces), grow geometrically upstream, stay fine over the
    first 10 d behind the slot (jet and recirculation), and grow towards the
    outflow. Normal to the wall the first cell is ``dy1`` and ``ny`` cells
    grow to the top. Markers: ``plate`` (no-slip wall), ``slot`` (jet exit),
    ``inlet``, ``outlet``, ``top``.
    """
    dx = 1.0 / n_slot
    slot_x = np.linspace(-0.5, 0.5, n_slot + 1)
    up = -0.5 - np.cumsum(_graded(x_up - 0.5, dx, 60))[::-1]
    near = 0.5 + np.cumsum(_graded(10.0, dx, 120))  # x = 0.5 .. 10.5
    far = near[-1] + np.cumsum(_graded(x_down - near[-1], near[-1] - near[-2], 60))
    xs = np.concatenate((up, slot_x, near, far))
    ys = np.concatenate(([0.0], np.cumsum(_graded(height, dy1, ny))))
    mesh = structured(xs, ys, names=("bottom", "outlet", "top", "inlet"))

    def slot(x, y):
        return np.abs(x) < 0.5

    return split_boundary(
        mesh, "bottom", {"slot": slot, "plate": lambda x, y: ~slot(x, y)}
    )


def plate_lm_mesh(nx_plate=120, ny=56):
    """T3 flat-plate mesh for the Langtry-Menter runs.

    Inlet 0.1 m upstream, plate 1.6 m, height 0.5 m, first cell 1e-5 m
    (y+ < 1 up to Re/m 1.3e6). Markers: inlet, outlet, top, symmetry
    (ahead of the plate), plate.
    """
    length, height, x0 = 1.6, 0.5, -0.1
    y1 = 1.0e-5
    r = geometric_growth(height, y1, ny)
    ys = np.concatenate(([0.0], np.cumsum(y1 * r ** np.arange(ny))))
    ys[-1] = height
    dx0 = 2.0e-3  # plate: dx 2 mm at the leading edge, growing geometrically
    g = geometric_growth(length, dx0, nx_plate)
    xp = np.concatenate(([0.0], np.cumsum(dx0 * g ** np.arange(nx_plate))))
    xp[-1] = length
    nxf = 12  # upstream: 12 cells, geometric toward the leading edge
    gf = geometric_growth(-x0, dx0, nxf)
    xf = -np.concatenate(([0.0], np.cumsum(dx0 * gf ** np.arange(nxf))))[::-1]
    xf[0] = x0
    xs = np.concatenate((xf[:-1], xp))
    mesh = structured(xs, ys)
    return split_boundary(
        mesh,
        "bottom",
        {"symmetry": lambda x, y: x < 0.0, "plate": lambda x, y: x >= 0.0},
    )


def naca0012_shedding_mesh(n_surface=384):
    """O-grid around the NACA 0012 for laminar vortex shedding (Re = 1000).

    Radial spacing (chord = 1): geometric from 0.002 at the wall up to 0.025,
    constant 0.025 out to 5 chords (wake region), then geometric growth to the
    farfield at 30 chords. The marching follows
    :func:`~pyaeroflow.mesh.airfoil.airfoil_ogrid` (smoothed normals blended
    into the radial direction) with this spacing instead of a single growth
    ratio. Boundary faces: airfoil, farfield.
    """
    first, growth, plateau, r_plateau, r_far = 0.002, 1.08, 0.025, 5.0, 30.0
    spacing, d = [], 0.0
    s = first
    while d < r_far:
        if d < r_plateau:
            s = min(s * growth, plateau) if spacing else first
        else:
            s = s * growth
        spacing.append(s)
        d += s
    spacing = np.array(spacing) * (r_far / d)  # land exactly on the farfield radius

    surface = airfoil_points(n_surface)
    n = surface.shape[0]
    n_radial = len(spacing)
    centre = 0.5 * (surface.min(axis=0) + surface.max(axis=0))
    chord = np.ptp(surface[:, 0])
    layers = [surface]
    normals = _smooth(_loop_normals(surface), 6)
    loop, distance = surface, 0.0
    for ds in spacing:
        radial = loop - centre
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        b = min(1.0, distance / (2.0 * chord))
        direction = (1.0 - b) * normals + b * radial
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        loop = loop + ds * direction
        distance += ds
        w = 0.25 * b
        loop = (1.0 - w) * loop + w * 0.5 * (
            np.roll(loop, 1, axis=0) + np.roll(loop, -1, axis=0)
        )
        layers.append(loop)
        normals = _smooth(_loop_normals(loop), 2 + int(20 * b))
    nodes = np.vstack(layers)

    def node(j, k):
        """Node index of radial layer j, surface point k (periodic)."""
        return j * n + (k % n)

    j, k = np.meshgrid(np.arange(n_radial), np.arange(n), indexing="ij")
    j, k = j.ravel(), k.ravel()
    quads = np.column_stack(
        (node(j, k), node(j + 1, k), node(j + 1, k + 1), node(j, k + 1))
    )
    _check_folds(nodes, quads)
    k = np.arange(n)
    boundary = {
        "airfoil": np.column_stack((node(0, k + 1), node(0, k))),
        "farfield": np.column_stack((node(n_radial, k), node(n_radial, k + 1))),
    }
    return Mesh.from_cells(nodes, quads, boundary)


def _channel_quad(nx=32, ny=16):
    return rectangle(
        nx, ny, x=(0.0, 2.0), y=(0.0, 1.0), cell_type="quad", cluster_y="both", beta=1.5
    )


def _channel_tri(nx=32, ny=16):
    return rectangle(
        nx,
        ny,
        x=(0.0, 2.0),
        y=(0.0, 1.0),
        cell_type="triangle",
        perturb=0.3,
        seed=1,
    )


def _cavity(n=64):
    return rectangle(n, n, cell_type="quad")


def _cavity_clustered(n=64, beta=1.5):
    return rectangle(
        n, n, cell_type="quad", cluster_x="both", cluster_y="both", beta=beta
    )


def _rayleigh_benard(nx=64, ny=32, beta=1.5, perturb=0.3, seed=3):
    """Periodic Rayleigh-Benard box: one critical wavelength 2 pi / 3.117 = 2.016.

    Unit plate distance, x from 0 to 2.016, cells clustered at the plates
    (tanh, ``beta``). Markers: ``bottom``, ``top`` (plates), ``left`` and
    ``right`` (periodic pair, translation 2.016 plate distances). The
    interior nodes are moved randomly by ``perturb`` of the local cell size
    (fixed ``seed``) so that the conduction state is not exact and the rolls
    start from the discretisation error. The conduction state (Nu = 1) is a
    steady solution too and the solver would stay on it from a regular mesh.
    """
    return rectangle(
        nx,
        ny,
        x=(0.0, 2.0 * np.pi / 3.117),
        y=(0.0, 1.0),
        cell_type="quad",
        cluster_y="both",
        beta=beta,
        perturb=perturb,
        seed=seed,
    )


def heated_plate_mesh(nx_plate=96, ny=48, length=1.0, height=0.3, upstream=0.25):
    """Laminar flat plate (heated-plate validation): quads, plate at y = 0.

    The plate runs from x = 0 to ``length`` with a symmetry strip of
    ``upstream`` ahead of it; the first cell is 2e-4 ``length`` high and the
    cells grow to ``height``. Cells along the plate are clustered towards the
    leading edge. Markers: inlet, outlet, top, symmetry, plate.
    """
    y1 = 2.0e-4 * length
    r = geometric_growth(height, y1, ny)
    ys = np.concatenate(([0.0], np.cumsum(y1 * r ** np.arange(ny))))
    ys[-1] = height
    dx0 = 2.0e-3 * length
    g = geometric_growth(length, dx0, nx_plate)
    xp = np.concatenate(([0.0], np.cumsum(dx0 * g ** np.arange(nx_plate))))
    xp[-1] = length
    nxf = 16
    gf = geometric_growth(upstream, dx0, nxf)
    xf = -np.concatenate(([0.0], np.cumsum(dx0 * gf ** np.arange(nxf))))[::-1]
    xf[0] = -upstream
    mesh = structured(
        np.concatenate((xf[:-1], xp)), ys, names=("bottom", "outlet", "top", "inlet")
    )
    mesh = split_boundary(
        mesh,
        "bottom",
        {"symmetry": lambda x, y: x < 0.0, "plate": lambda x, y: x >= 0.0},
    )
    return mesh


def _cylinder_ogrid(n_surface=128, n_radial=48, outer_radius=10.0):
    return annulus(
        n_surface,
        n_radial,
        radius=0.5,
        outer_radius=outer_radius,
        growth=1.0 + 6.0 / n_radial,
    )


def _plate(nx_plate=96, ny=64, reynolds=5.0e6):
    from pyaeroflow.cases.tmr_flat_plate import plate_mesh

    return plate_mesh(nx_plate, 24, ny, reynolds=reynolds)


def _ramp(nx=96, ny=64, angle=10.0):
    from pyaeroflow.cases.oblique_shock import ramp_mesh

    return ramp_mesh(nx, ny, angle=angle)


def _naca0012_euler_ogrid(n_surface=192, n_radial=64):
    return airfoil_ogrid(
        airfoil_points(n_surface, cut=1.0),
        n_radial,
        first_spacing=1.0e-3,
        farfield_radius=100.0,
    )


# name -> (builder, one-line description)
LIBRARY = {
    "channel-quad": (_channel_quad, "channel 2 x 1, quads clustered at both walls"),
    "channel-tri": (_channel_tri, "channel 2 x 1, perturbed triangles"),
    "cavity": (_cavity, "unit square, quads (lid-driven cavity)"),
    "cavity-clustered": (
        _cavity_clustered,
        "unit square, quads clustered at all walls (natural convection)",
    ),
    "rayleigh-benard": (
        _rayleigh_benard,
        "periodic box of one roll pair, plates clustered (Rayleigh-Benard)",
    ),
    "plate-heated": (
        heated_plate_mesh,
        "laminar flat plate with a symmetry strip ahead (heated plate)",
    ),
    "cylinder-ogrid": (_cylinder_ogrid, "O-grid around a cylinder, 10 radii"),
    "plate-rans": (
        _plate,
        "TMR flat-plate domain, y+ = 1 at the given Reynolds number",
    ),
    "plate-lm": (plate_lm_mesh, "T3 flat plate for the Langtry-Menter runs"),
    "ramp": (_ramp, "flat wall and compression ramp (Mach 2 Euler)"),
    "naca0012-ogrid": (naca0012_ogrid, "NACA 0012 O-grid, y+ from the Reynolds number"),
    "naca0012-euler-ogrid": (
        _naca0012_euler_ogrid,
        "NACA 0012 O-grid, farfield at 100 chords (compressible Euler)",
    ),
    "naca0012-shedding": (
        naca0012_shedding_mesh,
        "NACA 0012 O-grid for laminar shedding at Re 1000",
    ),
    "jet-crossflow": (jet_crossflow_mesh, "jet in crossflow over a flat plate, slot"),
}


def build(name, **parameters):
    """Build the library mesh ``name`` (parameters as the builder's keywords)."""
    if name not in LIBRARY:
        raise ValueError(f"unknown mesh '{name}'; known: {', '.join(sorted(LIBRARY))}")
    return LIBRARY[name][0](**parameters)
