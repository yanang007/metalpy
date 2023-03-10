import numpy as np
from numpy.testing import assert_equal

from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Cuboid, Prism, Ellipsoid, Tunnel, Obj2


def test_build():
    scene = Scene.of(
        Cuboid([1, 1, 1], size=2),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3).rotated(90, 0, 0, degrees=True),
        Ellipsoid.spheroid(1, 3, 0).translated(0, -2, 2),
        Tunnel([-3, 2, 2], 0.5, 1, 3),
        models=[1, 2, 3, 4, 5],
    )

    bounds = np.round(scene.bounds)
    ret = [
        scene.build(cell_size=0.2, bounds=bounds),
        scene.build(cell_size=[0.2, 0.2, 0.2], bounds=bounds),
        scene.build(n_cells=13500, bounds=bounds),
        scene.build(n_cells=[30, 30, 15], bounds=bounds)
    ]
    ref_mesh = ret[0]

    for model_mesh in ret[1:]:
        assert_equal(model_mesh.n_cells, ref_mesh.n_cells)
        assert_equal(model_mesh.base_mesh.cell_centers, ref_mesh.base_mesh.cell_centers)
        assert_equal(model_mesh.get_active_model(), model_mesh.get_active_model())
