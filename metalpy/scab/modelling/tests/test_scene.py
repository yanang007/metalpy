from numpy.testing import assert_equal

from metalpy.scab.modelling.scene import Scene
from metalpy.scab.modelling.shapes import Cuboid, Prism, Ellipsoid, Tunnel, Obj2


def test_build():
    scene = Scene.of(
        Cuboid([1, 1, 1], size=2),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3).rotated(90, 0, 0),
        Ellipsoid.spheroid(1, 3, 0).translated(0, -2, 2),
        Tunnel([-3, 2, 2], 0.5, 1, 3),
        models=[1, 2, 3, 4, 5],
    )

    mesh1, model1 = scene.build(grid_size=0.2)
    mesh2, model2 = scene.build(n_grids=[30, 30, 20])

    assert_equal(mesh1.n_cells, mesh2.n_cells)
    assert_equal(model1, model2)
