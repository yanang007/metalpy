from abc import ABC, abstractmethod

from ..transform import CompositeTransform, Transform, Translation, Rotation


class Shape3D(ABC):
    def __init__(self):
        self.transforms = CompositeTransform()
        pass

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

    @abstractmethod
    def clone(self):
        raise NotImplementedError()

    @abstractmethod
    def plot(self, ax, color):
        raise NotImplementedError()

    def apply(self, trans: Transform):
        """逻辑上对空间体位置进行变换，目前通过对网格点进行逆变换实现
        :param trans: 需要应用的变换
        :return:
        """
        self.transforms.add(trans)

    def translate(self, x, y, z):
        self.apply(Translation(-x, -y, -z))

    def rotate(self, y, a, b, degrees=True, seq='xyz'):
        self.apply(Rotation(-y, -a, -b, degrees=degrees, seq=seq))
