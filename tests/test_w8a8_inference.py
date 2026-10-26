"""Test released integer inference without including weight conversion code."""
import mlx.core as mx
import numpy as np
import pytest
from hi3d_mac.mlx_backend import w8a8


@pytest.mark.parametrize("rows,columns,width", [(3, 17, 64), (65, 131, 2112), (7, 33, 8448)])
def test_integer_projection_matches_numpy(rows, columns, width):
    rng = np.random.default_rng(7)
    x = mx.array(rng.normal(size=(rows, width)).astype(np.float16))
    w = mx.array(rng.integers(-127, 128, size=(columns, width), dtype=np.int8))
    sw = mx.array(rng.uniform(.0001, .001, columns).astype(np.float32))
    bias = mx.array(rng.normal(0, .01, columns).astype(np.float16))
    a, sa = w8a8.quantize_activation(x)
    integer = np.asarray(a, np.int64) @ np.asarray(w, np.int64).T
    expected = integer.astype(np.float32) * np.asarray(sa)[:, None] * np.asarray(sw)[None, :] + np.asarray(bias, np.float32)
    result = np.asarray(w8a8.linear(x, w, sw, bias), np.float32)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, rtol=5e-4, atol=2e-6)
