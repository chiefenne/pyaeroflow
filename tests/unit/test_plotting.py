"""Optional mesh plot (skipped without matplotlib)."""

import pytest

from pyaeroflow.mesh import rectangle

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_plot_mesh(tmp_path):
    import matplotlib.pyplot as plt

    from pyaeroflow.plotting import plot_mesh

    mesh = rectangle(4, 3, cell_type="mixed")
    ax = plot_mesh(mesh)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["bottom", "right", "top", "left"]
    ax.figure.savefig(tmp_path / "mesh.png")
    plt.close(ax.figure)
    assert (tmp_path / "mesh.png").stat().st_size > 0
