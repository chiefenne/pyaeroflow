"""Mesh check for airfoil cases (``pyaeroflow check-mesh``).

Reads the numbers that decide whether a RANS airfoil mesh is usable, from
``docs/pyaero.md`` section 2: units, first layer thickness and y⁺, points
on the surface, farfield distance and abrupt changes of the cell size.
Invalid meshes (bow-tie cells, faces without a boundary set) are already
rejected by the readers.
"""

from dataclasses import dataclass, field

import numpy as np

from pyaeroflow.mesh.ordering import order_boundary_faces

PYAERO_WALL = "airfoil"


@dataclass
class MeshReport:
    """Facts and findings of :func:`check_mesh`."""

    facts: list = field(default_factory=list)
    findings: list = field(default_factory=list)  # (level, text)
    wall: str | None = None
    chord: float | None = None
    y_plus_max: float | None = None

    def add(self, key, value):
        """Record a fact."""
        self.facts.append((key, value))

    def warn(self, text):
        """Record a finding that needs attention."""
        self.findings.append(("warning", text))

    def note(self, text):
        """Record a hint."""
        self.findings.append(("note", text))

    @property
    def n_warnings(self):
        """Return the number of warnings."""
        return sum(1 for level, _ in self.findings if level == "warning")

    def format(self):
        """Return the report as text."""
        lines = [f"{k:<24}{v}" for k, v in self.facts]
        lines.append("")
        if not self.findings:
            lines.append("no findings")
        for level, text in self.findings:
            lines.append(f"{level}: {text}")
        return "\n".join(lines)


def flat_plate_cf(reynolds):
    """Turbulent flat-plate C_f at mid-chord, the estimate behind pyaero.md's table."""
    return 0.0592 * (0.5 * reynolds) ** -0.2


def first_cell_for_y_plus(reynolds, y_plus=1.0):
    """First-cell *height* y1 / c for ``y_plus`` at the cell centre."""
    return 2.0 * y_plus / (reynolds * np.sqrt(0.5 * flat_plate_cf(reynolds)))


def check_mesh(mesh, wall=None, reynolds=None, chord=None):
    """Check a mesh for a wall-resolved airfoil case.

    Parameters
    ----------
    mesh : Mesh
        Mesh in metres (after ``scale``).
    wall : str, optional
        Name of the airfoil boundary; default ``"airfoil"`` (PyAero).
    reynolds : float, optional
        Chord Reynolds number; enables the y⁺ estimate.
    chord : float, optional
        Chord in metres; default the x extent of the wall.

    Returns
    -------
    MeshReport
    """
    rep = MeshReport()
    n_nodes = np.bincount(mesh.cell_n_nodes, minlength=5)
    rep.add("cells", f"{mesh.n_cells}  ({n_nodes[3]} triangles, {n_nodes[4]} quads)")
    rep.add("nodes, faces", f"{mesh.n_nodes}, {mesh.n_faces}")
    for name, faces in mesh.boundary_faces.items():
        ctr = mesh.face_centre[faces]
        rep.add(
            f"boundary {name}",
            f"{faces.size} faces, x {ctr[:, 0].min():.4g}..{ctr[:, 0].max():.4g}, "
            f"y {ctr[:, 1].min():.4g}..{ctr[:, 1].max():.4g}",
        )
    if mesh.n_cells > 400_000:
        rep.note(
            f"{mesh.n_cells} cells: expect about 30 s per Newton iteration on 8 cores"
        )

    wall = wall or (PYAERO_WALL if PYAERO_WALL in mesh.boundary_faces else None)
    if wall is None:
        rep.warn(
            f"no boundary called '{PYAERO_WALL}'; name the wall with --wall "
            f"(boundaries: {', '.join(mesh.boundary_faces)})"
        )
        return rep
    if wall not in mesh.boundary_faces:
        rep.warn(f"boundary '{wall}' not found ({', '.join(mesh.boundary_faces)})")
        return rep
    rep.wall = wall
    faces = mesh.boundary_faces[wall]
    wall_nodes = mesh.nodes[np.unique(mesh.face_nodes[faces])]
    x_min, x_max = wall_nodes[:, 0].min(), wall_nodes[:, 0].max()
    chord = float(chord) if chord else float(x_max - x_min)
    rep.chord = chord
    rep.add("wall", f"'{wall}', chord {chord:.6g} m (x {x_min:.4g}..{x_max:.4g})")
    if not 0.05 <= chord <= 20.0:
        rep.warn(
            f"chord {chord:.4g} looks like a different length unit; the mesh must "
            "be in metres (mesh.scale = 0.001 for millimetres)"
        )

    _first_layer(mesh, faces, rep, reynolds, chord)
    _surface_points(mesh, faces, rep)
    _farfield(mesh, wall, rep, chord, x_min)
    _size_jumps(mesh, rep)
    return rep


def _first_layer(mesh, faces, rep, reynolds, chord):
    owner = mesh.owner[faces]
    normal = mesh.face_normal[faces]
    d = np.abs(
        np.einsum(
            "fk,fk->f", mesh.cell_centroid[owner] - mesh.face_centre[faces], normal
        )
    )
    rep.add(
        "first cell height y1",
        f"{2 * d.min():.3e} .. {2 * d.max():.3e} m "
        f"(y1/c {2 * d.min() / chord:.2e} .. {2 * d.max() / chord:.2e}), "
        f"median {2 * np.median(d) / chord:.2e} c",
    )
    if reynolds is None:
        rep.note("add --re to estimate y+ and the first-cell height it needs")
        return
    cf = flat_plate_cf(reynolds)
    y_plus = d / chord * reynolds * np.sqrt(0.5 * cf)
    rep.y_plus_max = float(y_plus.max())
    rep.add(
        "Re, Cf estimate", f"{reynolds:.4g}, {cf:.5f} (turbulent flat plate, x/c = 0.5)"
    )
    rep.add("y+ (cell centre)", f"{y_plus.min():.2f} .. {y_plus.max():.2f}")
    need = first_cell_for_y_plus(reynolds, 1.0)
    rep.add("y1/c for y+ = 1", f"{need:.2e}")
    if y_plus.max() > 3.0:
        rep.warn(
            f"y+ up to {y_plus.max():.1f}: the first layer is too thick for a "
            f"wall-resolved model (a y+ of 12 gave 40 % too little skin friction); "
            f"use y1/c of about {need:.1e} or less at Re {reynolds:.3g}"
        )
    elif y_plus.max() > 1.5:
        rep.note(
            f"y+ up to {y_plus.max():.1f} by the flat-plate estimate; the suction "
            "peak has a higher C_f, check the y_plus column of the wall CSV after a run"
        )
    if y_plus.max() < 0.1:
        rep.note("y+ below 0.1: finer than needed, the cell count may be avoidable")


def _surface_points(mesh, faces, rep):
    order = order_boundary_faces(mesh.face_nodes[faces], mesh.nodes)
    if order is None:
        rep.warn("the wall faces do not form one single contour")
        return
    ctr = mesh.face_centre[faces][order]
    le = int(np.argmin(ctr[:, 0]))
    upper, lower = le, faces.size - le
    rep.add("surface faces", f"{faces.size} ({upper} upper, {lower} lower)")
    few = min(upper, lower)
    if few < 100:
        rep.warn(
            f"{few} faces on one side: too few to resolve the pressure "
            "distribution (about 100 per side give the lift within 1 %)"
        )
    elif few < 400:
        rep.note(
            f"{few} faces on the smaller side: fine for lift; for the drag within "
            "a few percent use 400-500 per side with clustering at both edges"
        )


def _farfield(mesh, wall, rep, chord, x_le):
    ref = np.array(
        [x_le + 0.25 * chord, mesh.face_centre[mesh.boundary_faces[wall]][:, 1].mean()]
    )
    shortest = np.inf
    for name, faces in mesh.boundary_faces.items():
        if name == wall:
            continue
        dist = np.hypot(*(mesh.face_centre[faces] - ref).T) / chord
        rep.add(f"distance to {name}", f"{dist.min():.1f} .. {dist.max():.1f} chords")
        shortest = min(shortest, dist.min())
    if shortest < 10.0:
        rep.warn(
            f"farfield only {shortest:.1f} chords away: the lift is biased (3.5 chords "
            "gave -6 %, -3 % with the vortex correction); use 20-50 chords and "
            "reference.vortex_correction = true"
        )
    elif shortest < 20.0:
        rep.note(
            f"farfield {shortest:.1f} chords away; with reference.vortex_correction "
            "= true this is acceptable, 20-50 chords is the recommendation"
        )


def _size_jumps(mesh, rep):
    inner = mesh.neighbour >= 0
    a = mesh.cell_area[mesh.owner[inner]]
    b = mesh.cell_area[mesh.neighbour[inner]]
    ratio = np.maximum(a / b, b / a)
    worst = int(np.argmax(ratio))
    where = mesh.face_centre[inner][worst]
    rep.add(
        "cell area ratio (faces)",
        f"p99 {np.percentile(ratio, 99):.2f}, max {ratio.max():.1f} "
        f"at ({where[0]:.3g}, {where[1]:.3g})",
    )
    if ratio.max() > 20.0:
        rep.warn(
            f"abrupt cell size change (area ratio {ratio.max():.0f}) at "
            f"({where[0]:.3g}, {where[1]:.3g}); look at the mesh there"
        )
    elif ratio.max() > 10.0:
        rep.note(
            f"area ratio {ratio.max():.0f} between neighbouring cells at "
            f"({where[0]:.3g}, {where[1]:.3g}), usually where mesh blocks meet"
        )
