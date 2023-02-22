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

    ret = [
        scene.build(cell_size=0.2),
        scene.build(cell_size=[0.2, 0.2, 0.2]),
        scene.build(n_cells=18000),
        scene.build(n_cells=[30, 30, 20])
    ]
    ref_mesh, ref_model = ret[0]

    for mesh, model in ret[1:]:
        assert_equal(mesh.n_cells, ref_mesh.n_cells)
        assert_equal(mesh.cell_centers, ref_mesh.cell_centers)
        assert_equal(model, ref_model)
