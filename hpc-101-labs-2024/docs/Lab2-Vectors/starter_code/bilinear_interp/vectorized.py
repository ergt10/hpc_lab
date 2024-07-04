import numpy as np
from numpy import int64


def bilinear_interp_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This is the vectorized implementation of bilinear interpolation.
    - a is a ND array with shape [N, H1, W1, C], dtype = int64
    - b is a ND array with shape [N, H2, W2, 2], dtype = float64
    - return a ND array with shape [N, H2, W2, C], dtype = int64
    """
    # get axis size from ndarray shape
    N, H1, W1, C = a.shape
    N1, H2, W2, _ = b.shape
    assert N == N1

    # TODO: Implement vectorized bilinear interpolation
    x = b[..., 0]
    y = b[..., 1]
    x0_idx = np.floor(x).astype(int)
    y0_idx = np.floor(y).astype(int)

    x1_idx = x0_idx + 1
    y1_idx = y0_idx + 1   
    _x = x - x0_idx
    _y = y - y0_idx
    Ia = a[np.arange(N)[:, None, None], x0_idx, y0_idx]
    Ib = a[np.arange(N)[:, None, None], x1_idx, y0_idx]
    Ic = a[np.arange(N)[:, None, None], x0_idx, y1_idx]
    Id = a[np.arange(N)[:, None, None], x1_idx, y1_idx] 
    wa = (1 - _x) * (1 - _y)
    wb = _x * (1 - _y)
    wc = (1 - _x) * _y
    wd = _x * _y
    res = (Ia * wa[..., None] +
           Ib * wb[..., None] +
           Ic * wc[..., None] +
           Id * wd[..., None]).astype(np.int64)

    return res
    