import numpy as np
import pytest
import tifffile


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def write_tif(path, array):
    tifffile.imwrite(str(path), array)
    return path


@pytest.fixture
def tif_writer():
    return write_tif


def make_blob_mask(shape, center, radius, value=1):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    mask = np.zeros(shape, dtype=np.uint8)
    mask[disk] = value
    return mask


@pytest.fixture
def blob_mask():
    return make_blob_mask
