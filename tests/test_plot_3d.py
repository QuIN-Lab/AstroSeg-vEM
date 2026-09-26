import numpy as np
import pytest

from reconstruction.plot_3d import (
    compute_auto_camera,
    faces_to_pyvista,
    load_mesh,
    load_mesh_npz,
    load_mesh_ply,
    parse_camera_position,
    parse_scale_factors,
    parse_vector3,
)
from reconstruction.reconstruct_3d import save_mesh_npz, save_mesh_ply


def test_parse_scale_factors():
    assert np.allclose(parse_scale_factors("1,2,3"), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        parse_scale_factors("1,2")


def test_parse_camera_position_valid():
    pos = parse_camera_position("1,2,3; 0,0,0; 0,0,1")
    assert pos == ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def test_parse_camera_position_none_when_empty():
    assert parse_camera_position("") is None


def test_parse_camera_position_wrong_triplet_count_raises():
    with pytest.raises(ValueError):
        parse_camera_position("1,2,3; 0,0,0")


def test_parse_vector3():
    v = parse_vector3("1, 1, 1", "camera_direction")
    assert np.allclose(v, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        parse_vector3("1,1", "camera_direction")


def test_faces_to_pyvista_prepends_triangle_count():
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    pv_faces = faces_to_pyvista(faces)
    assert pv_faces.shape == (2, 4)
    assert (pv_faces[:, 0] == 3).all()
    assert np.array_equal(pv_faces[:, 1:], faces)


def test_compute_auto_camera_faces_center_from_direction():
    verts = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32)
    pos, focal, _up = compute_auto_camera(verts, direction=np.array([1, 0, 0]), up=np.array([0, 0, 1]),
                                         distance_factor=1.0)
    assert focal == pytest.approx((5.0, 5.0, 5.0))

    assert pos[0] > focal[0]
    assert pos[1] == pytest.approx(focal[1])
    assert pos[2] == pytest.approx(focal[2])


def test_compute_auto_camera_rejects_zero_direction():
    verts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    with pytest.raises(ValueError):
        compute_auto_camera(verts, direction=np.zeros(3), up=np.array([0, 0, 1]), distance_factor=1.0)


def test_load_mesh_npz_roundtrip(tmp_path):
    verts = np.random.default_rng(0).random((6, 3)).astype(np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    path = tmp_path / "mesh.npz"
    save_mesh_npz(path, verts, faces, scale=np.ones(3), offset=np.zeros(3))

    loaded_verts, loaded_faces, meta = load_mesh_npz(path)
    assert np.allclose(loaded_verts, verts)
    assert np.array_equal(loaded_faces, faces)
    assert "scale" in meta and "offset" in meta


def test_load_mesh_ply_roundtrip_recovers_offset(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    path = tmp_path / "mesh.ply"
    save_mesh_ply(path, verts, faces, offset=np.array([1.0, 2.0, 3.0]))

    loaded_verts, loaded_faces, meta = load_mesh_ply(path)
    assert loaded_verts.shape == verts.shape
    assert loaded_faces.shape == faces.shape
    assert np.allclose(meta["offset"], [1.0, 2.0, 3.0])


def test_load_mesh_dispatches_by_suffix(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    npz_path = tmp_path / "mesh.npz"
    save_mesh_npz(npz_path, verts, faces, scale=np.ones(3), offset=np.zeros(3))

    v, _f, _meta = load_mesh(npz_path)
    assert np.allclose(v, verts)


def test_load_mesh_unsupported_suffix_raises(tmp_path):
    path = tmp_path / "mesh.obj"
    path.write_text("not a real mesh")
    with pytest.raises(ValueError):
        load_mesh(path)
