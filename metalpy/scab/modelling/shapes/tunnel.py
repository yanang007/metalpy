import numpy as np

from . import Shape3D
from .bounds import Bounds


def is_abs_distance_in(arr, x0, r):
    return np.abs(arr - x0) < r


class Tunnel(Shape3D):
    def __init__(self, p0, r0, r1, L):
        """定义一个起始点在x0，内外圆半径r0和r1，向x方向延申长度为L的空心圆柱环
        :param p0: 起始点，隧道的左中位置
        :param r0: 内圆半径
        :param r1: 外圆半径
        :param L: 隧道长度
        """
        super().__init__()
        self.p0 = p0 = np.asarray(p0)
        self.r0 = r0
        self.r1 = r1
        self.L = L

    def do_place(self, mesh_cell_centers, worker_id):
        x0, y0, z0 = self.p0
        L = self.L
        r0, r1 = self.r0, self.r1
        xs, ys, zs = mesh_cell_centers[:, 0], mesh_cell_centers[:, 1], mesh_cell_centers[:, 2]
        # 判定是否在长L，宽高r1的长方体范围内
        included = \
            is_abs_distance_in(xs, x0 + L / 2, L / 2) & \
            is_abs_distance_in(ys, y0, r1) & \
            is_abs_distance_in(zs, z0, r1)

        indices = included

        possible_grids = mesh_cell_centers[included, :]  # 只计算在这个长方体区域内的方块是否属于圆环柱
        rs = np.sqrt((possible_grids[:, 1] - self.p0[1]) ** 2 + (possible_grids[:, 2] - self.p0[2]) ** 2)
        indices[included] = (rs < r1) & (rs > r0)

        return indices

    def do_hash(self):
        return hash((*self.p0, self.r0, self.r1, self.L))

    def do_clone(self):
        return Tunnel(self.p0, self.r0, self.r1, self.L)

    @property
    def local_bounds(self):
        x0, y0, z0 = self.p0
        r1 = self.r1
        L = self.L
        return Bounds(*np.r_[
            x0, L,
            y0 - r1, y0 + r1,
            z0 - r1, z0 + r1,
        ].ravel())

    def to_local_polydata(self):
        import pyvista as pv

        resolution = 100
        a = np.linspace(0, 2 * np.pi, resolution + 1)[:-1]  # 2π位置和0位置重复，去掉最后一个
        x0, y0, z0 = self.p0
        r0, r1 = self.r0, self.r1
        L = self.L
        x = np.r_[x0, x0 + L]
        y = np.r_[y0 + r0 * np.cos(a), y0 + r1 * np.cos(a)]
        z = np.r_[z0 + r0 * np.sin(a), z0 + r1 * np.sin(a)]

        # array(resolution, ) [<x0内圆...>, <x0外圆...>, <x0+L内圆...>, <x0+L外圆...>]
        xs = x.repeat(2 * resolution)
        ys = np.tile(y, 2)
        zs = np.tile(z, 2)

        indices = np.arange(xs.shape[0]).reshape([4, -1])
        edge_counts = np.ones(resolution, dtype=np.integer) * 4
        bottom_faces = np.c_[edge_counts, indices[0], indices[1], np.roll(indices[1], -1), np.roll(indices[0], -1)]
        top_faces = np.c_[edge_counts, indices[2], indices[3], np.roll(indices[3], -1), np.roll(indices[2], -1)]
        outer_side_faces = np.c_[edge_counts, indices[1], indices[3], np.roll(indices[3], -1), np.roll(indices[1], -1)]
        inner_side_faces = np.c_[edge_counts, indices[0], indices[2], np.roll(indices[2], -1), np.roll(indices[0], -1)]

        faces = np.c_[bottom_faces, top_faces, outer_side_faces, inner_side_faces].ravel()

        return pv.PolyData(np.c_[xs, ys, zs], faces=faces)
