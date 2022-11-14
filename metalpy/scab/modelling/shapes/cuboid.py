import numpy as np

from . import Shape3D
from metalpy.utils.misc import plot_opaque_cube


def is_inside_cuboid(mesh, corner, lengths):
    deltas = mesh - corner
    return np.all(deltas >= 0, axis=1) & np.all(deltas <= lengths, axis=1)


class Cuboid(Shape3D):
    def __init__(self, p1, p2):
        super().__init__()
        self.p1, self.p2 = p1, p2 = np.asarray(p1), np.asarray(p2)
        self.corner = np.min(np.c_[p1, p2], axis=1)
        self.lengths = np.abs(p2 - p1)

    def do_place(self, mesh_cell_centers, worker_id):
        indices = is_inside_cuboid(mesh_cell_centers, self.corner, self.lengths) # np.all((mesh.cell_centers - self.corner) < self.lengths, axis=1)
        return indices

    @property
    def x0(self): return self.corner[0]

    @property
    def x1(self): return self.corner[0] + self.lengths[0]

    @property
    def y0(self): return self.corner[1]

    @property
    def y1(self): return self.corner[1] + self.lengths[1]

    @property
    def z0(self): return self.corner[2]

    @property
    def z1(self): return self.corner[2] + self.lengths[2]

    @property
    def direction(self):
        """
        获取横梁最长轴方向
        :return: 0, 1, 2代表x, y, z
        """
        return np.argmax(self.lengths)

    def markHeight(self, mesh2d):
        corner2d = self.corner[0:2]
        lengths2d = self.lengths[0:2]
        indices = is_inside_cuboid(mesh2d, corner2d, lengths2d)

        indices = indices * self.z1()

        return indices

    def __hash__(self):
        return hash((*self.corner, *self.lengths))

    def do_clone(self):
        return Cuboid(self.p1, self.p2)

    def plot(self, ax, color):
        plot_opaque_cube(ax, *self.corner, *self.lengths, color=color)

    @property
    def local_bounds(self):
        return np.c_[self.corner, self.corner + self.lengths].ravel()
