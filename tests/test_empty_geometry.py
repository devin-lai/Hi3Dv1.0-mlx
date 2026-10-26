import numpy as np
import pytest
import torch
from hi3d_mac.fast_unique import unique_keys
from hi3d_mac.lean_extract import construct_voxel_grid


@pytest.mark.parametrize("counts", [False, True])
def test_unique_empty_keys_matches_numpy(counts):
    keys = np.empty(0, dtype=np.int64)
    values, inverse, observed_counts = unique_keys(keys, return_counts=counts)
    reference = np.unique(keys, return_inverse=True, return_counts=counts)
    np.testing.assert_array_equal(values, reference[0])
    np.testing.assert_array_equal(inverse, reference[1])
    if counts:
        np.testing.assert_array_equal(observed_counts, reference[2])
    else:
        assert observed_counts is None


def test_empty_voxel_grid_preserves_shapes_and_dtypes():
    vertices, cubes = construct_voxel_grid(torch.empty((0, 3), dtype=torch.int32))
    assert vertices.shape == (0, 3) and vertices.dtype == torch.int32
    assert cubes.shape == (0, 8) and cubes.dtype == torch.int64
