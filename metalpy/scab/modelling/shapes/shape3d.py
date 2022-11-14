from abc import ABC, abstractmethod

import numpy as np

from ..transform import CompositeTransform, Transform, Translation, Rotation


class Shape3D(ABC):
    def __init__(self):
        self.transforms = CompositeTransform()

    def place(self, mesh_cell_centers, worker_id):
        """
        注意：调用者会占用tqdm的position=0位置，如果继承的place方法内有自己定义的进度条，
        需要特别指定position为其他位置
        :param mesh_cell_centers: 3维网格中心点坐标
        :param worker_id: worker的id，一般用于指示是否显示进度条
        :return: mask[n_cells]，指示每个网格是否有效的
        """
        mesh = self.before_place(mesh_cell_centers, worker_id)
        return self.do_place(mesh, worker_id)

    def before_place(self, mesh_cell_centers, worker_id):
        return self.transforms.inverse_transform(mesh_cell_centers)

    @abstractmethod
    def do_place(self, mesh_cell_centers, worker_id):
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    def clone(self):
        ret = self.do_clone()
        ret.transforms = self.transforms.clone()
        return ret

    @abstractmethod
    def do_clone(self):
        raise NotImplementedError()

    @abstractmethod
    def plot(self, ax, color):
        raise NotImplementedError()

    @property
    def center(self):
        return self.transforms.transform(self.local_center)

    @property
    def bounds(self):
        bounds = self.oriented_bounds
        ret = np.zeros(6)
        ret[::2] = np.min(bounds, axis=1)
        ret[1::2] = np.max(bounds, axis=1)

        return ret

    @property
    def oriented_bounds(self):
        bounds = self.local_oriented_bounds
        return self.transforms.transform(bounds).T

    @property
    def local_center(self):
        """返回模型在局部坐标系下的中心点
        :return: array(3) [x, y, z]
        """
        bounds = self.local_oriented_bounds
        return bounds.mean(axis=0)

    @property
    def local_bounds(self):
        """返回模型在局部坐标系下的矩形包围盒
        基类需要至少实现local_oriented_bounds和local_bounds中的一个
        :return: array(6) [x0, x1, y0, y1, z0, z1]
        """
        bounds = self.local_oriented_bounds
        ret = np.zeros(6)
        ret[::2] = np.min(bounds, axis=1)
        ret[1::2] = np.max(bounds, axis=1)

        return ret

    @property
    def local_oriented_bounds(self):
        """返回模型在局部坐标系下的八点包围盒
        基类需要至少实现local_oriented_bounds和local_bounds中的一个
        :return: array(8, 3) [[x0, y0, z0], ...[x7, y7, z7]]
        """
        local_bounds = self.local_bounds
        x, y, z = np.meshgrid(local_bounds[0:2], local_bounds[2:4], local_bounds[4:6], indexing='ij')
        return np.c_[x.ravel(), y.ravel(), z.ravel()]

    @property
    def local_bounding_size(self):
        bounds = self.local_bounds
        return bounds[1::2] - bounds[::2]

    def apply(self, trans: Transform, inplace=False):
        """逻辑上对空间体位置进行变换，目前通过对网格点进行逆变换实现
        :param trans: 需要应用的变换
        :param inplace: 操作是否应用在当前实例上
        :return: 当inplace为True，返回当前实例，否则返回一个变换后的新对象
        """
        if inplace:
            ret = self
        else:
            ret = self.clone()

        ret.transforms.add(trans)
        return ret

    def translate(self, x, y, z, inplace=False):
        return self.apply(Translation(x, y, z), inplace=inplace)

    def rotate(self, y, a, b, degrees=True, seq='xyz', inplace=False):
        return self.apply(Rotation(y, a, b, degrees=degrees, seq=seq), inplace=inplace)

    def translated(self, x, y, z, inplace=True):
        return self.translate(x, y, z, inplace=inplace)

    def rotated(self, y, a, b, degrees=True, seq='xyz', inplace=True):
        return self.rotate(y, a, b, degrees=degrees, seq=seq, inplace=inplace)


def bounding_box_of(shapes: list[Shape3D]):
    bounds = None
    for m in shapes:
        if bounds is None:
            bounds = np.asarray(m.bounds)
            continue
        bounds[1::2] = np.max([bounds[1::2], m.bounds[1::2]], axis=0)
        bounds[0::2] = np.min([bounds[0::2], m.bounds[0::2]], axis=0)

    return bounds
