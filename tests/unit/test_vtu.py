"""VTU writer: structure, round trip, and (if installed) reading with VTK."""

import base64
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pyaeroflow.io import write_vtu
from pyaeroflow.mesh import rectangle

_DTYPES = {"Float64": np.float64, "Int64": np.int64, "UInt8": np.uint8}


def decode(element):
    """Decode a DataArray written by write_vtu."""
    dtype = _DTYPES[element.get("type")]
    text = element.text.strip()
    if element.get("format") == "ascii":
        values = np.array(text.split(), dtype=dtype)
    else:
        raw = base64.b64decode(text)
        n_bytes = int(np.frombuffer(raw[:8], dtype=np.uint64)[0])
        values = np.frombuffer(raw[8 : 8 + n_bytes], dtype=dtype)
    return values.reshape(-1, int(element.get("NumberOfComponents")))


@pytest.fixture
def mesh():
    return rectangle(5, 4, cell_type="polygon", perturb=0.3, seed=7)


@pytest.fixture
def fields(mesh):
    cell = {"area": mesh.cell_area, "centroid": mesh.cell_centroid}
    point = {"x": mesh.nodes[:, 0]}
    return cell, point


@pytest.mark.parametrize("binary", [True, False])
def test_round_trip(tmp_path, mesh, fields, binary):
    cell, point = fields
    path = write_vtu(tmp_path / "m.vtu", mesh, cell, point, binary=binary)
    root = ET.parse(path).getroot()
    arrays = {a.get("Name"): decode(a) for a in root.iter("DataArray")}
    np.testing.assert_array_equal(arrays["Points"][:, :2], mesh.nodes)
    np.testing.assert_array_equal(arrays["connectivity"].ravel(), mesh.cell_nodes)
    np.testing.assert_array_equal(arrays["offsets"].ravel(), mesh.cell_ptr[1:])
    np.testing.assert_array_equal(arrays["area"].ravel(), mesh.cell_area)
    np.testing.assert_array_equal(arrays["centroid"][:, :2], mesh.cell_centroid)
    np.testing.assert_array_equal(arrays["x"].ravel(), mesh.nodes[:, 0])
    types = arrays["types"].ravel()
    assert set(types) <= {5, 7, 9}


def test_wrong_length_rejected(tmp_path, mesh):
    with pytest.raises(ValueError, match="expected"):
        write_vtu(tmp_path / "m.vtu", mesh, {"bad": np.zeros(3)})


@pytest.mark.parametrize("binary", [True, False])
def test_readable_by_vtk(tmp_path, mesh, fields, binary):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    cell, point = fields
    path = write_vtu(tmp_path / "m.vtu", mesh, cell, point, binary=binary)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfPoints() == mesh.n_nodes
    assert grid.GetNumberOfCells() == mesh.n_cells
    area = vtk_to_numpy(grid.GetCellData().GetArray("area"))
    np.testing.assert_array_equal(area, mesh.cell_area)
    centroid = vtk_to_numpy(grid.GetCellData().GetArray("centroid"))
    np.testing.assert_array_equal(centroid[:, :2], mesh.cell_centroid)
