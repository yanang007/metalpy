import numpy as np

from metalpy.utils.dhash import dhash
from . import Shape3D
from .bounds import NoBounds


class FullSpace(Shape3D):
    def __init__(self):
        """定义一个覆盖整个空间的几何体
        """
        super().__init__()

    def do_place(self, mesh_cell_centers, worker_id):
        indices = np.ones(mesh_cell_centers.shape[0], dtype=bool)
        return indices

    def do_hash(self):
        return 0

    def __dhash__(self):
        return dhash()

    def do_clone(self):
        return FullSpace()

    @property
    def local_bounds(self):
        return NoBounds()

    def to_local_polydata(self):
        return None
