import numpy as np
import pytest

from reconstruction.postprocess import (
    build_center_lookup,
    build_centers_from_labels,
    collect_topk_dataframe,
    compute_topk_labels_memmap,
    downsample_binary_mask,
    fill_holes_2d,
    filter_cc_by_center,
    make_disk_structure,
    read_pred_with_image_mask,
    resolve_image_path,
    run_hdbscan,
    scale_single_slice_params_for_downsample,
    strip_ds_suffix,
    suppress_single_slice_artifacts_in_memory,
    upsample_binary_mask,
)


def test_strip_ds_suffix():
    assert strip_ds_suffix("dataset_4_mutant_dsxy10", "_dsxy10") == "dataset_4_mutant"
    assert strip_ds_suffix("dataset_4_mutant", "_dsxy10") == "dataset_4_mutant"
    assert strip_ds_suffix("dataset_4_mutant", "") == "dataset_4_mutant"


def test_make_disk_structure_is_symmetric_and_centered():
    struct = make_disk_structure(3)
    assert struct.shape == (7, 7)
    assert struct[3, 3]
    assert np.array_equal(struct, struct[::-1, :])
    assert np.array_equal(struct, struct[:, ::-1])


def test_make_disk_structure_zero_radius_is_single_pixel():
    assert make_disk_structure(0).shape == (1, 1)


def test_scale_single_slice_params_for_downsample():
    assert scale_single_slice_params_for_downsample(2000, 12, downsample_xy=1) == (2000, 12)

    assert scale_single_slice_params_for_downsample(2000, 10, downsample_xy=10) == (20, 1)


def test_downsample_upsample_binary_mask_roundtrip_shape():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    small = downsample_binary_mask(mask, downsample_xy=10)
    assert small.shape == (4, 4)
    back = upsample_binary_mask(small, mask.shape)
    assert back.shape == mask.shape

    assert back[15:25, 15:25].mean() > 0.8


def test_resolve_image_path_matches_by_stripped_suffix(tmp_path, tif_writer):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    tif_writer(image_dir / "image_001.tif", np.zeros((4, 4), dtype=np.uint8))
    found = resolve_image_path(str(tmp_path / "image_001_pred_2d.tif"), str(image_dir), "_pred_2d")
    assert found == str(image_dir / "image_001.tif")


def test_resolve_image_path_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_image_path(str(tmp_path / "image_999_pred_2d.tif"), str(tmp_path), "_pred_2d")


def test_read_pred_with_image_mask_zeroes_out_background(tmp_path, tif_writer):
    pred = np.ones((8, 8), dtype=np.uint8)
    img = np.ones((8, 8), dtype=np.uint8)
    img[0:2, 0:2] = 0

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    tif_writer(image_dir / "image_001.tif", img)
    pred_path = tif_writer(tmp_path / "image_001_pred_2d.tif", pred)

    masked = read_pred_with_image_mask(str(pred_path), image_dir=str(image_dir))
    assert (masked[0:2, 0:2] == 0).all()
    assert (masked[2:, 2:] == 1).all()


def test_collect_topk_dataframe_ranks_by_area(tmp_path, tif_writer, blob_mask):
    shape = (64, 64)
    big = blob_mask(shape, center=(20, 20), radius=10)
    small = blob_mask(shape, center=(50, 50), radius=3)
    pred = np.where(big | small, 1, 0).astype(np.uint8)
    path = tif_writer(tmp_path / "image_000_pred_2d.tif", pred)

    df = collect_topk_dataframe([str(path)], top_k=2)
    assert len(df) == 2
    rank1 = df[df["rank"] == 1].iloc[0]
    rank2 = df[df["rank"] == 2].iloc[0]
    assert rank1["area_pixels"] > rank2["area_pixels"]
    assert rank1["centroid_y"] == pytest.approx(20, abs=1)
    assert rank1["centroid_x"] == pytest.approx(20, abs=1)


def test_collect_topk_dataframe_pads_missing_ranks_with_nan(tmp_path, tif_writer, blob_mask):
    pred = blob_mask((32, 32), center=(16, 16), radius=5)
    path = tif_writer(tmp_path / "image_000_pred_2d.tif", pred)
    df = collect_topk_dataframe([str(path)], top_k=3)
    assert len(df) == 3
    assert np.isnan(df.iloc[2]["centroid_y"])
    assert df.iloc[2]["area_pixels"] == 0


def _soma_track_df(blob_mask, shape, n_slices, soma_center, soma_radius):
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for z in range(n_slices):
        rows.append({
            "slice_index": z, "slice_file": f"slice_{z:03d}.tif", "rank": 1,
            "area_pixels": int(np.pi * soma_radius ** 2),
            "centroid_y": float(soma_center[0]) + rng.normal(0, 0.5),
            "centroid_x": float(soma_center[1]) + rng.normal(0, 0.5),
        })
    return pd.DataFrame(rows)


def test_run_hdbscan_finds_single_tight_cluster(tmp_path, blob_mask):
    df = _soma_track_df(blob_mask, (64, 64), n_slices=10, soma_center=(30, 30), soma_radius=6)
    out_json = tmp_path / "soma.json"
    payload = run_hdbscan(df, str(out_json), z_scale=1.0, min_cluster_size=3, min_samples=2)
    assert payload["status"] == "ok"
    assert payload["num_clusters"] >= 1
    best = payload["best_cluster"]
    assert best["center_y_weighted"] == pytest.approx(30, abs=1)
    assert best["center_x_weighted"] == pytest.approx(30, abs=1)
    assert out_json.exists()


def test_run_hdbscan_raises_on_all_nan_rows(tmp_path):
    import pandas as pd
    df = pd.DataFrame([{"slice_index": 0, "rank": 1, "area_pixels": 5,
                        "centroid_y": np.nan, "centroid_x": np.nan}])
    with pytest.raises(ValueError):
        run_hdbscan(df, str(tmp_path / "out.json"))


def test_build_center_lookup_roundtrips_run_hdbscan_output(tmp_path, blob_mask):
    df = _soma_track_df(blob_mask, (64, 64), n_slices=8, soma_center=(12, 40), soma_radius=5)
    out_json = tmp_path / "soma.json"
    run_hdbscan(df, str(out_json), z_scale=1.0, min_cluster_size=3, min_samples=2)
    global_center, lookup = build_center_lookup(str(out_json))
    assert global_center == pytest.approx((12, 40), abs=1)
    assert len(lookup) > 0
    for cy, cx in lookup.values():
        assert (cy, cx) == pytest.approx((12, 40), abs=1)


def test_filter_cc_by_center_removes_distant_large_artifact_keeps_soma(tmp_path, tif_writer, blob_mask):


    shape = (100, 100)
    soma = blob_mask(shape, center=(20, 20), radius=8)
    artifact = blob_mask(shape, center=(80, 80), radius=10)
    pred = np.where(soma | artifact, 1, 0).astype(np.uint8)
    path = tif_writer(tmp_path / "image_000_pred_2d.tif", pred)

    out_dir = tmp_path / "filtered"
    filter_cc_by_center(
        [str(path)], str(out_dir),
        global_center=(20, 20), center_lookup={0: (20, 20)},
        rel_area_ratio=0.5, rel_dist_ratio=0.1,
    )

    import tifffile
    out = tifffile.imread(str(out_dir / "image_000_pred_2d.tif"))
    assert out[20, 20] == 1
    assert out[80, 80] == 0


def test_filter_cc_by_center_single_target_protect_keeps_something(tmp_path, tif_writer, blob_mask):


    shape = (100, 100)
    only_component = blob_mask(shape, center=(90, 90), radius=10)
    path = tif_writer(tmp_path / "image_000_pred_2d.tif", only_component)

    out_dir = tmp_path / "filtered"
    filter_cc_by_center(
        [str(path)], str(out_dir),
        global_center=(0, 0), center_lookup={},
        rel_area_ratio=0.0, rel_dist_ratio=0.0,
        single_target_protect=True,
    )
    import tifffile
    out = tifffile.imread(str(out_dir / "image_000_pred_2d.tif"))
    assert out.sum() > 0


def test_suppress_single_slice_artifacts_removes_unsupported_outlier():
    shape = (30, 30)
    n_slices = 5
    volume = np.zeros((n_slices, *shape), dtype=np.uint8)

    volume[:, 5:9, 5:9] = 1

    volume[2, 15:29, 15:29] = 1

    cleaned = suppress_single_slice_artifacts_in_memory(
        volume, area_ratio=1.5, support_radius=2, min_area=10,
    )
    assert cleaned[2, 20, 20] == 0
    assert cleaned[:, 6, 6].all()


def test_suppress_single_slice_artifacts_leaves_supported_growth_alone():
    shape = (20, 20)
    volume = np.zeros((3, *shape), dtype=np.uint8)
    volume[0, 5:8, 5:8] = 1
    volume[1, 4:9, 4:9] = 1
    volume[2, 5:8, 5:8] = 1
    cleaned = suppress_single_slice_artifacts_in_memory(
        volume, area_ratio=1.5, support_radius=3, min_area=10,
    )
    assert cleaned[1].sum() == volume[1].sum()


def test_suppress_single_slice_artifacts_rejects_non_3d_volume():
    with pytest.raises(ValueError):
        suppress_single_slice_artifacts_in_memory(np.zeros((10, 10)))


def test_compute_topk_labels_and_build_centers_keep_only_largest(tmp_path, blob_mask):
    shape = (24, 24)
    n_slices = 4
    volume = np.zeros((n_slices, *shape), dtype=np.uint8)
    big = blob_mask(shape, center=(6, 6), radius=4)
    small = blob_mask(shape, center=(18, 18), radius=2)
    for z in range(n_slices):
        volume[z] = np.where(big | small, 1, 0)

    labels_mem, keep_labels, mask_path, labels_path = compute_topk_labels_memmap(
        volume, topk=1, level=0.5, work_dir=tmp_path,
    )
    assert len(keep_labels) == 1
    kept_label = next(iter(keep_labels))

    assert labels_mem[0, 6, 6] == kept_label
    assert labels_mem[0, 18, 18] != kept_label

    centers = build_centers_from_labels(labels_mem, keep_labels, ds_xy=1, ds_z=1, orig_z=n_slices)
    assert set(centers.keys()) == set(range(n_slices))
    for z in range(n_slices):
        (cy, cx), = centers[z]
        assert (cy, cx) == pytest.approx((6, 6), abs=1)

    import os
    os.unlink(mask_path)
    os.unlink(labels_path)


def test_fill_holes_2d_fills_a_ring(tmp_path, tif_writer):
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:16, 4:16] = 1
    mask[8:12, 8:12] = 0
    assert mask[10, 10] == 0

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    tif_writer(in_dir / "s0_pred_2d.tif", mask)

    out_dir = tmp_path / "out"
    fill_holes_2d(str(in_dir), str(out_dir), "*_pred_2d.tif", "uint8")

    import tifffile
    filled = tifffile.imread(str(out_dir / "s0_pred_2d.tif"))
    assert filled[10, 10] == 1
    assert filled[4, 4] == 1
    assert filled[0, 0] == 0


def test_downsample_and_upsample_mask_dir_roundtrip_shape(tmp_path, tif_writer):
    from reconstruction.postprocess import downsample_mask_dir, upsample_mask_dir

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    tif_writer(in_dir / "s0.tif", mask)

    ds_dir = tmp_path / "ds"
    downsample_mask_dir(str(in_dir), str(ds_dir), "*.tif", "uint8", downsample_xy=10)

    import tifffile
    small = tifffile.imread(str(ds_dir / "s0.tif"))
    assert small.shape == (4, 4)

    up_dir = tmp_path / "up"
    upsample_mask_dir(str(ds_dir), str(up_dir), "*.tif", "uint8", out_shape=(40, 40))
    big = tifffile.imread(str(up_dir / "s0.tif"))
    assert big.shape == (40, 40)
