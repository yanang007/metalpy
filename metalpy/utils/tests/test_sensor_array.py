import numpy as np
from numpy.testing import assert_almost_equal

from metalpy.utils.sensor_array import get_grids_ex


def test_get_grids_ex():
    arrs = [
        # xs, ys, zs = [0, 1, 2, 3], [0, 1, 2, 3], [0]
        get_grids_ex(xs=np.arange(4), ys=np.arange(4), zs=0).pts,
        get_grids_ex(cell_width=[1, 1], nz=1, origin=[0, 0, 0], end=[3, 3, 3]).pts,
        get_grids_ex(n=4, nz=1, origin=[0, 0, 0], end=[3, 3, 3]).pts,
        get_grids_ex(n=4, nz=1, cell_width=[1, 1], origin=[0, 0, 0]).pts,
        get_grids_ex(n=4, nz=1, cell_width=[1, 1], end=[3, 3, 0]).pts,
    ]

    for arr in arrs:
        assert_almost_equal(arrs[0], arr)

    arr2 = get_grids_ex(n=4, cell_width=[1, 1], end=[3, 3, 0]).pts
    assert arr2.ndim == 2
