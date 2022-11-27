import numpy as np
import pytest

from metalpy.scab.modelling.shapes import Cuboid


def test_cuboid():
    # 测试各种定义方式的等价性
    cubes = [
        Cuboid([0, 2, 0], [2, 0, 2]),
        Cuboid([0, 0, 0], [2, 2, 2]),
        Cuboid(corner=[0, 0, 0], size=[2, 2, 2]),
        Cuboid(corner=[0, 0, 0], size=2),
        Cuboid(corner=[0, 0, 0], center=[1, 1, 1]),
        Cuboid(corner2=[2, 2, 2], size=2),
        Cuboid(corner2=[2, 2, 2], center=[1, 1, 1]),
        Cuboid(center=[1, 1, 1], size=[2, 2, 2]),
        Cuboid(bounds=[0, 2, 0, 2, 0, 2]),
    ]

    corners = np.vstack([cube.corner for cube in cubes])
    lengths = np.vstack([cube.lengths for cube in cubes])

    assert np.all(corners == corners[0])
    assert np.all(lengths == lengths[0])


def test_cuboid_failure():
    # 测试严格模式下错误定义报错
    with pytest.raises(ValueError):
        cube = Cuboid([0, 2, 0], [2, 0, 2], no_corner_adjust=True)
