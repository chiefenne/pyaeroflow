"""Case templates: ready-to-run case files with their meshes (``pyaeroflow init``).

Each template is one of the example cases of ``examples/cases`` (copied into
the package, so it works after a plain ``pip install``) together with the
library mesh it runs on (:mod:`pyaeroflow.mesh.library`). ``write_template``
writes the case file and builds the mesh next to it. The ``airfoil``
template is the exception: it needs the user's airfoil mesh and is written by
:func:`pyaeroflow.initcase.make_case` (``pyaeroflow init airfoil --mesh ...``,
the same as ``pyaeroflow init-case``).
"""

import re
from importlib import resources
from pathlib import Path
from typing import NamedTuple

from pyaeroflow.mesh import write_su2
from pyaeroflow.mesh.library import build


class Template(NamedTuple):
    """A case template.

    Attributes
    ----------
    group : str
        Heading of the listing (channel, plate, airfoil, ...).
    case_file : str or None
        Packaged TOML (None: the airfoil template, made by ``init-case``).
    mesh : str or None
        Library mesh name.
    mesh_parameters : dict
        Keywords of the library builder.
    mesh_file : str or None
        File name the case expects next to it (its ``mesh.file``).
    """

    group: str
    case_file: str | None
    mesh: str | None
    mesh_parameters: dict
    mesh_file: str | None


TEMPLATES = {
    "couette": Template(
        "channel", "couette.toml", "channel-quad", {}, "channel_quad_32x16.su2"
    ),
    "couette-unsteady": Template(
        "channel", "couette_unsteady.toml", "channel-quad", {}, "channel_quad_32x16.su2"
    ),
    "poiseuille": Template(
        "channel", "poiseuille.toml", "channel-tri", {}, "channel_tri_32x16.su2"
    ),
    "cavity": Template(
        "channel", "cavity_re100.toml", "cavity", {}, "cavity_quad_64.su2"
    ),
    "cavity-heated": Template(
        "channel",
        "cavity_heated_ra1e5.toml",
        "cavity-clustered",
        {"n": 64},
        "cavity_clustered_64.su2",
    ),
    "plate-heated-sa": Template(
        "plate",
        "plate_heated_sa.toml",
        "plate-rans",
        {"reynolds": 5.0e6},
        "plate_rans_96x64.su2",
    ),
    "plate-sa": Template(
        "plate",
        "flat_plate_rans.toml",
        "plate-rans",
        {"reynolds": 5.0e6},
        "plate_rans_96x64.su2",
    ),
    "plate-bcm": Template(
        "plate",
        "flat_plate_transition.toml",
        "plate-rans",
        {"reynolds": 5.0e5},
        "plate_transition_96x64.su2",
    ),
    "plate-lm-t3a": Template(
        "plate", "lm_t3a.toml", "plate-lm", {}, "plate_lm_120x56.su2"
    ),
    "plate-lm-t3am": Template(
        "plate", "lm_t3am.toml", "plate-lm", {}, "plate_lm_120x56.su2"
    ),
    "plate-lm-t3b": Template(
        "plate", "lm_t3b.toml", "plate-lm", {}, "plate_lm_120x56.su2"
    ),
    "jet-crossflow": Template(
        "jet", "jet_crossflow.toml", "jet-crossflow", {}, "jet_crossflow.su2"
    ),
    "jet-crossflow-inclined": Template(
        "jet", "jet_crossflow_inclined.toml", "jet-crossflow", {}, "jet_crossflow.su2"
    ),
    "cylinder-euler": Template(
        "bluff body",
        "cylinder_euler.toml",
        "cylinder-ogrid",
        {},
        "cylinder_ogrid_128x48.su2",
    ),
    "naca0012-shedding": Template(
        "bluff body",
        "naca0012_re1000_a20_spinup.toml",
        "naca0012-shedding",
        {},
        "naca0012_shedding.su2",
    ),
    "naca0012-sa": Template(
        "airfoil",
        "naca0012_rans.toml",
        "naca0012-ogrid",
        {},
        "naca0012_ogrid_256x128.su2",
    ),
    "naca0012-euler-m05": Template(
        "airfoil",
        "naca0012_euler_m05.toml",
        "naca0012-euler-ogrid",
        {},
        "naca0012_euler_ogrid_192x64.su2",
    ),
    "naca0012-euler-m08": Template(
        "airfoil",
        "naca0012_euler_m08.toml",
        "naca0012-euler-ogrid",
        {},
        "naca0012_euler_ogrid_192x64.su2",
    ),
    "ramp-m2": Template(
        "supersonic",
        "ramp_10deg_m2.toml",
        "ramp",
        {"nx": 96, "ny": 64},
        "ramp_10deg_96x64.su2",
    ),
    "airfoil": Template("airfoil", None, None, {}, None),
}

AIRFOIL_TEXT = "your airfoil mesh (PyAero .su2 or Gmsh .msh): --mesh, --model, --re"


def template_text(name):
    """Return the packaged case-file text of template ``name``."""
    t = _template(name)
    if t.case_file is None:
        raise ValueError(f"template '{name}' is made from a mesh: use --mesh")
    return resources.files(__name__).joinpath(t.case_file).read_text()


def description(name, width=72):
    """One-line description: the first sentence of the case file's header."""
    t = _template(name)
    if t.case_file is None:
        return AIRFOIL_TEXT
    words = []
    for line in template_text(name).splitlines():
        if not line.startswith("#"):
            break
        words.append(line.lstrip("#").strip())
    text = " ".join(words)
    text = re.sub(r"^[A-Z]\d+:\s*", "", text)  # drop the requirement tags (V1:, R2:)
    text = re.split(r"(?<=[a-z0-9)])\.\s", text + " ", maxsplit=1)[0].rstrip(". ")
    return text if len(text) <= width else text[: width - 3].rstrip() + "..."


def case_name(text):
    """Return the ``[case] name`` of a case-file text."""
    match = re.search(r'^\[case\]\s*\nname\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else None


def write_template(name, directory=".", case=None, force=False):
    """Write template ``name`` into ``directory``: the case file and its mesh.

    Parameters
    ----------
    name : str
        Key of :data:`TEMPLATES` (not ``airfoil``).
    directory : str or Path
        Created if missing.
    case : str, optional
        New case name (file name, ``[case] name`` and output directory).
    force : bool
        Overwrite an existing case file. An existing mesh file is kept unless
        ``force`` is set (several templates share one mesh).

    Returns
    -------
    case_path, mesh_path : Path
    mesh_written : bool
    """
    t = _template(name)
    text = template_text(name)
    old = case_name(text)
    new = case or old
    if case and case != old:
        text = re.sub(
            r'^(\[case\]\s*\nname\s*=\s*)"[^"]+"',
            rf'\1"{new}"',
            text,
            count=1,
            flags=re.M,
        )
        text = text.replace(f'"output/{old}"', f'"output/{new}"')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    case_path = directory / f"{new}.toml"
    if case_path.exists() and not force:
        raise FileExistsError(f"{case_path} exists (use --force to overwrite)")
    header = (
        f"# Written by 'pyaeroflow init {name}' (template of examples/cases/"
        f"{t.case_file}); mesh: pyaeroflow make-mesh {t.mesh}"
        + "".join(f" --set {k}={v:g}" for k, v in t.mesh_parameters.items())
        + f" -o {t.mesh_file}\n"
    )
    case_path.write_text(header + text)
    mesh_path = directory / t.mesh_file
    written = False
    if force or not mesh_path.exists():
        write_su2(mesh_path, build(t.mesh, **t.mesh_parameters))
        written = True
    return case_path, mesh_path, written


def _template(name):
    if name not in TEMPLATES:
        raise ValueError(
            f"unknown template '{name}'; known: {', '.join(sorted(TEMPLATES))}"
        )
    return TEMPLATES[name]
