import numpy as np

from scipy.stats import linregress

from metalpy.scab.modelling.shapes.cuboid import is_inside_cuboid
from metalpy.utils.dhash import dhash
from metalpy.utils.ear_clip import ear_clip
from . import Shape3D
from .bounds import Bounds


def is_abs_distance_in(arr, x0, r):
    return np.abs(arr - x0) < r


def is_in_multipoints_or_point(pt, mp_or_pt):
    if mp_or_pt.geom_type == 'Point':
        return pt == mp_or_pt
    else:
        return pt in mp_or_pt.geoms


class Edge:
    def __init__(self, p1, p2):
        self.p1, self.p2 = p1, p2
        self.ymin = min(p1[1], p2[1])
        self.ymax = max(p1[1], p2[1])
        self.xmin = min(p1[0], p2[0])
        self.xmax = max(p1[0], p2[0])

        ends = np.vstack([p1, p2])

        if p1[0] != p2[0]:  # 非垂直于x轴
            slope, intercept, _, _, _ = linregress(ends[:, 0], ends[:, 1])
            self.k, self.b = slope, intercept
        else:
            self.k = None
            self.x0 = p1[0]

    def intersects(self, pts):
        x, y = pts[:, 0], pts[:, 1]
        if self.k is None:
            # 处理与垂直线相交的情形
            # 由于是多面体，因此与该直线相交的射线一定与其重合
            # 其所在直线会与该边的相邻两个边也存在交点
            # 交于下端点视为相交，其余位置不视为相交
            # 可以保证边缘点被视为多边形内
            # 1. 点p在该线段之下，则只会与相邻边有2个交点
            # 2. 点p在该线段与下方邻边交点，则会与相邻边和该边有3个交点
            # 3. 点p在该线段中间及上方邻边交点，则只会与上方邻边有1个交点
            indices = x == self.x0
            indices[indices] = y[indices] == self.ymin
            return indices
        elif self.k == 0:
            return (x >= self.xmin) & (x <= self.xmax) & (y <= self.ymin)
        else:
            y_intersection = self.k * x + self.b
            return (y_intersection >= self.ymin) & (y_intersection <= self.ymax) & (y <= y_intersection)


class Prism(Shape3D):
    def __init__(self, pts, z0, z1, verbose=True):
        """定义任意底面棱锥

        Parameters
        ----------
        pts
            底面多边形顶点列表
        z0, z1
            底面和顶面高度
        verbose
            是否输出辅助信息（主要为顶点数较多时导出PyVista模型会触发进度条）
        """
        super().__init__()
        self.pts = np.asarray(pts)
        self.z0 = min(z0, z1)
        self.z1 = max(z0, z1)
        self.verbose = verbose

    def do_place(self, mesh_cell_centers, worker_id):
        # 优化: 只判断在xyz三轴边界框内的点
        p0 = np.asarray((*np.min(self.pts, axis=0), self.z0))
        p1 = np.asarray((*np.max(self.pts, axis=0), self.z1))
        indices = is_inside_cuboid(mesh_cell_centers, p0, p1 - p0)

        n_possible_grids = np.sum(indices)

        # 优化: 使用射线法，取向+y方向的射线
        if n_possible_grids > 0:
            edges = []
            n_edges = len(self.pts)
            for i in range(n_edges):
                edges.append(Edge(self.pts[i], self.pts[(i + 1) % n_edges]))

            mesh = mesh_cell_centers[indices, 0:2]
            n_intersects = np.zeros(n_possible_grids, dtype=int)

            for edge in edges:
                intercepts = edge.intersects(mesh)
                n_intersects = n_intersects + intercepts

            indices_horizontally_satisfied = (n_intersects & 1) == 1
        else:
            indices_horizontally_satisfied = False

        indices[indices] = indices_horizontally_satisfied

        return indices

    def do_hash(self):
        return hash((*self.pts.ravel(), self.z0, self.z1))

    def __dhash__(self):
        return dhash(super().__dhash__(),
                     self.z0, self.z1, *self.pts.ravel())

    def do_clone(self):
        return Prism(self.pts, self.z0, self.z1)

    @property
    def local_bounds(self):
        return Bounds(*np.c_[
            np.r_[self.pts.min(axis=0), self.z0],
            np.r_[self.pts.max(axis=0), self.z1]
        ].ravel())

    def triangulated_vertices(self):
        return ear_clip(self.pts, verbose=self.verbose)

    def to_local_polydata(self):
        import pyvista as pv

        pts = self.pts
        z0, z1 = self.z0, self.z1
        n_pts = len(pts)
        n_vertices = 2 * len(pts)
        vh = pts.repeat(2, axis=0)
        vz = np.asarray([[z0, z1]]).repeat(n_pts, axis=0).ravel()
        vertices = np.c_[vh, vz]
        indices = np.arange(0, n_vertices, 1).reshape([-1, 2]).T

        faces = self.triangulated_vertices()
        top_face = np.c_[np.repeat(3, faces.shape[0]), faces * 2].ravel()  # 上顶面
        bottom_face = np.c_[np.repeat(3, faces.shape[0]), faces * 2 + 1].ravel()  # 下底面
        edge_counts = np.ones(n_pts, dtype=np.integer) * 4
        side_faces = np.c_[
            edge_counts,
            indices[0], indices[1],
            np.roll(indices[1], -1), np.roll(indices[0], -1)
        ].ravel()

        faces = np.r_[top_face, bottom_face, side_faces]

        shape = pv.PolyData(vertices.astype(np.float64), faces=faces)
        return shape
