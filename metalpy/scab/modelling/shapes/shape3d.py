from abc import ABC, abstractmethod

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
        return self.transforms.transform(mesh_cell_centers)

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
        return self.apply(Translation(-x, -y, -z), inplace=inplace)

    def rotate(self, y, a, b, degrees=True, seq='xyz', inplace=False):
        return self.apply(Rotation(-y, -a, -b, degrees=degrees, seq=seq), inplace=inplace)

    def translated(self, x, y, z, inplace=False):
        return self.translate(x, y, z, inplace=inplace)

    def rotated(self, y, a, b, degrees=True, seq='xyz', inplace=False):
        return self.rotated(y, a, b, degrees=degrees, seq=seq, inplace=inplace)
