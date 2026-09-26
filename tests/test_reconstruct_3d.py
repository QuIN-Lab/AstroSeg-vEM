import numpy as np
import pytest
import tifffile
from PIL import Image

from reconstruction.reconstruct_3d import (
    compute_integer_axis_config,
    crop_png_background,
    keep_largest_mesh_component,
    load_and_downsample_volume_streaming,
    marching_cubes_pymcubes,
    pad_png_to_canvas,
    parse_scale_factors,
    save_mesh_npz,
    save_mesh_ply,
    save_params_json,
    save_volume_tif,
)


def test_parse_scale_factors_valid():
    factors = parse_scale_factors("20, 10, 10")
    assert np.allclose(factors, [20.0, 10.0, 10.0])


def test_parse_scale_factors_wrong_count_raises():
    with pytest.raises(ValueError):
        parse_scale_factors("1,2")


def test_compute_integer_axis_config_covers_data_range():
    verts = np.array([[0, 0, 0], [10, 20, 30]], dtype=np.float32)
    cfg = compute_integer_axis_config(verts)
    x0, x1, y0, y1, z0, z1 = cfg["axes_ranges"]
    assert x0 <= 0 and x1 >= 10
    assert y0 <= 20 and y1 >= 20
    assert z0 <= 30 and z1 >= 30
    assert cfg["n_ylabels"] == 5
    assert cfg["n_zlabels"] == 5


def _write_slices(dir_path, shape, n_slices, fg_value=1.0):
    for i in range(n_slices):
        arr = np.zeros(shape, dtype=np.float32)
        arr[shape[0] // 4: shape[0] * 3 // 4, shape[1] // 4: shape[1] * 3 // 4] = fg_value
        tifffile.imwrite(str(dir_path / f"slice_{i:03d}.tif"), arr)


def test_load_and_downsample_volume_streaming_basic(tmp_path):
    _write_slices(tmp_path, (40, 40), n_slices=6)
    volume, files, raw_shape = load_and_downsample_volume_streaming(tmp_path, pattern="*.tif")
    assert raw_shape == (6, 40, 40)
    assert volume.shape == (6, 40, 40)
    assert len(files) == 6


def test_load_and_downsample_volume_streaming_downsamples_xy(tmp_path):
    _write_slices(tmp_path, (40, 40), n_slices=4)
    volume, _files, _raw_shape = load_and_downsample_volume_streaming(tmp_path, pattern="*.tif", downsample_xy=4)
    assert volume.shape == (4, 10, 10)


def test_load_and_downsample_volume_streaming_z_subsample(tmp_path):
    _write_slices(tmp_path, (16, 16), n_slices=10)
    volume, _files, _raw_shape = load_and_downsample_volume_streaming(tmp_path, pattern="*.tif", downsample_z=3)
    assert volume.shape[0] == len(range(0, 10, 3))


def test_load_and_downsample_volume_streaming_skip_first_slices(tmp_path):
    _write_slices(tmp_path, (16, 16), n_slices=5)
    volume, _files, _raw_shape = load_and_downsample_volume_streaming(
        tmp_path, pattern="*.tif", skip_first_slices=2,
    )
    assert volume.shape[0] == 3


def test_load_and_downsample_volume_streaming_rejects_inconsistent_shapes(tmp_path):
    tifffile.imwrite(str(tmp_path / "a.tif"), np.zeros((16, 16), dtype=np.float32))
    tifffile.imwrite(str(tmp_path / "b.tif"), np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        load_and_downsample_volume_streaming(tmp_path, pattern="*.tif")


def test_load_and_downsample_volume_streaming_no_files_raises(tmp_path):
    with pytest.raises(ValueError):
        load_and_downsample_volume_streaming(tmp_path, pattern="*.tif")


def test_marching_cubes_on_filled_cube_produces_mesh():
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[4:12, 4:12, 4:12] = 1.0
    verts, faces = marching_cubes_pymcubes(volume, level=0.5)
    assert len(verts) > 0
    assert len(faces) > 0
    assert faces.dtype == np.int32

    assert faces.max() < len(verts)


def test_marching_cubes_empty_volume_produces_no_mesh():
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    verts, faces = marching_cubes_pymcubes(volume, level=0.5)
    assert len(verts) == 0
    assert len(faces) == 0


def test_keep_largest_mesh_component_drops_small_island():

    verts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [10, 10, 10], [11, 10, 10], [10, 11, 10],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    new_verts, new_faces = keep_largest_mesh_component(verts, faces)
    assert len(new_verts) == 3
    assert len(new_faces) == 1
    assert np.allclose(new_verts, verts[:3])


def test_save_and_reload_mesh_npz(tmp_path):
    verts = np.random.default_rng(0).random((5, 3)).astype(np.float32)
    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
    path = tmp_path / "mesh.npz"
    save_mesh_npz(path, verts, faces, scale=np.array([1, 1, 1]), offset=np.array([0, 0, 0]))

    data = np.load(path)
    assert np.allclose(data["verts"], verts)
    assert np.array_equal(data["faces"], faces)


def test_save_mesh_ply_has_expected_header_and_counts(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    path = tmp_path / "mesh.ply"
    save_mesh_ply(path, verts, faces, offset=np.array([1.0, 2.0, 3.0]))

    text = path.read_text()
    assert "element vertex 3" in text
    assert "element face 1" in text
    assert "comment offset_z 1.00" in text


def test_save_volume_tif_roundtrip(tmp_path):
    volume = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
    path = tmp_path / "volume.tif"
    save_volume_tif(path, volume)
    reloaded = tifffile.imread(str(path))
    assert np.allclose(reloaded, volume)


def test_save_params_json_creates_parents_and_content(tmp_path):
    path = tmp_path / "nested" / "params.json"
    save_params_json(path, {"a": 1})
    assert path.exists()
    import json
    assert json.loads(path.read_text()) == {"a": 1}


def test_crop_png_background_shrinks_to_foreground(tmp_path):
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[20:30, 20:30] = (255, 255, 255)
    path = tmp_path / "img.png"
    Image.fromarray(img).save(path)

    info = crop_png_background(path, background_rgb=(0, 0, 0), padding=2)
    assert info["cropped_size"][0] < info["original_size"][0]
    with Image.open(path) as reloaded:
        assert reloaded.size == tuple(info["cropped_size"])


def test_crop_png_background_all_background_is_a_noop(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    path = tmp_path / "blank.png"
    Image.fromarray(img).save(path)
    info = crop_png_background(path, background_rgb=(0, 0, 0), padding=2)
    assert info["cropped_size"] == info["original_size"]


def test_pad_png_to_canvas_centers_image(tmp_path):
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    path = tmp_path / "small.png"
    Image.fromarray(img).save(path)

    info = pad_png_to_canvas(path, canvas_size=(20, 20), background_rgb=(0, 0, 0))
    assert info["paste_xy"] == [5, 5]
    with Image.open(path) as reloaded:
        assert reloaded.size == (20, 20)


def test_pad_png_to_canvas_raises_if_image_too_big(tmp_path):
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    path = tmp_path / "big.png"
    Image.fromarray(img).save(path)
    with pytest.raises(ValueError):
        pad_png_to_canvas(path, canvas_size=(10, 10))
