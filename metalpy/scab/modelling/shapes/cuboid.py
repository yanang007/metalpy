from typing import Iterable, Union

import numpy as np

from . import Shape3D
from metalpy.utils.misc import plot_opaque_cube


def is_inside_cuboid(mesh, corner, lengths):
    deltas = mesh - corner
    return np.all(deltas >= 0, axis=1) & np.all(deltas <= lengths, axis=1)


class Cuboid(Shape3D):
    def __init__(self,
                 corner: Iterable = None,
                 corner2: Iterable = None,
                 size: Union[Iterable, float] = None,
                 center: Iterable = None,
                 bounds: Iterable = None,
                 no_corner_adjust=False):
        """通过立方体的中心、长度、角点中任意两个参数或单独使用边界来定义立方体

        Parameters
        ----------
        corner
            立方体的(-x, -y, -z)方向角点坐标
        corner2
            立方体的(+x, +y, +z)方向角点坐标
        size
            立方体的长宽高
        center
            中心点坐标
        bounds
            立方体的边界，当定义时，无视其他参数，[x0, x1, y0, y1, z0, z1]
        no_corner_adjust
            是否拒绝调整角点坐标，默认为假，出现负数边长时，会自动调整角点坐标，若为真，则抛出异常
        """
        super().__init__()

        if bounds is None and np.r_[
            corner is not None,
            corner2 is not None,
            center is not None,
            size is not None,
        ].sum() != 2:
            raise ValueError("Bounds or exactly two of center, lengths, corner, corner2 must be specified.")

        if size is not None:
            size = np.asarray(size)
            if size.size == 1:
                size = np.ones(3) * size

        if bounds is not None:
            corner = np.array(bounds[::2])
            size = np.array(bounds[1::2]) - corner
        elif corner is not None:  # 和corner2或center，计算size
            corner = np.asarray(corner)
            if corner2 is not None:
                size = np.asarray(corner2) - np.asarray(corner)
            elif center is not None:
                size = 2 * (np.asarray(center) - np.asarray(corner))
        elif corner2 is not None:  # 和center或size，计算corner
            if center is not None:
                size = 2 * (np.asarray(corner2) - np.asarray(center))
                corner = np.asarray(corner2) - size
            elif size is not None:
                corner = np.asarray(corner2) - np.asarray(size)
        elif size is not None:  # 和center，计算corner
            if center is not None:
                corner = np.asarray(center) - size / 2

        if np.any(size < 0):
            if no_corner_adjust:
                raise ValueError("Negative length detected.")
            # 纠正负数边长
            corner[size < 0] += size[size < 0]
            size = np.abs(size)

        self.corner = corner
        self.lengths = size

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

    def do_hash(self):
        return hash((*self.corner, *self.lengths))

    def do_clone(self):
        return Cuboid(corner=self.corner, size=self.lengths)

    def plot(self, ax, color):
        plot_opaque_cube(ax, *self.corner, *self.lengths, color=color)

    @property
    def local_bounds(self):
        return np.c_[self.corner, self.corner + self.lengths].ravel()

    def to_local_polydata(self):
        import pyvista as pv

        return pv.Cube(bounds=self.local_bounds)
