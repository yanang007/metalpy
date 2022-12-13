import numpy as np

from . import Shape3D
from .cuboid import is_inside_cuboid
from .bounds import Bounds


class Ellipsoid(Shape3D):
    def __init__(self, a, b, c):
        """定义一个椭球体，球心在原点

        Parameters
        ----------
        a
            椭球体在x方向上的半轴长
        b
            椭球体在y方向上的半轴长
        c
            椭球体在z方向上的半轴长
        """
        super().__init__()
        self.radii = np.asarray([a, b, c])

    @staticmethod
    def sphere(r):
        """定义一个球体

        Parameters
        ----------
        r
            半径

        Returns
        -------
            半径为r的球体
        """
        return Ellipsoid(r, r, r)

    @staticmethod
    def spheroid(a, c, polar_axis=2):
        """定义一个旋转椭球体（三个轴长为a, a, c）

        a > c时，其为扁椭球体

        a < c时，其为长椭球体

        默认polar_axis = 2时，其极轴在z轴上，方程为
            (x^2 + y^2) / a^2 + z^2 / c^2 = 1

        Parameters
        ----------
        a
            equatorial，半赤道轴长
        c
            polar，半极轴长
        polar_axis
            极轴方向，0 - x，1 - y，2 - z

        Returns
        -------
            中心点在原点，半赤道轴长为a，半极轴长为c，极轴为polar_axis的旋转椭球体
        """
        r = [a, a, a]
        r[polar_axis] = c
        return Ellipsoid(*r)

    def do_place(self, mesh_cell_centers, worker_id):
        # 优化：只判断在xyz三轴边界框内的点
        possible_targets = is_inside_cuboid(mesh_cell_centers, -self.radii, 2 * self.radii)

        indices = np.zeros(len(mesh_cell_centers), dtype=np.bool8)
        indices[possible_targets] = np.sum((mesh_cell_centers[possible_targets] / self.radii) ** 2, axis=1) < 1

        return indices

    def do_hash(self):
        return hash((*self.radii,))

    def do_clone(self):
        return Ellipsoid(*self.radii)

    @property
    def local_bounds(self):
        return Bounds(*np.c_[-self.radii, self.radii].ravel())

    def to_local_polydata(self):
        import pyvista as pv

        return pv.ParametricEllipsoid(*self.radii)
