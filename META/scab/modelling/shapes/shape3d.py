from abc import ABC, abstractmethod


class Shape3D(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def place(self, mesh_cell_centers, worker_id):
        """
        注意：调用者会占用tqdm的position=0位置，如果继承的place方法内有自己定义的进度条，
        需要特别指定position为其他位置
        :param mesh_cell_centers: 3维网格中心点坐标
        :param worker_id: worker的id，一般用于指示是否显示进度条
        :return: mask[n_cells]，指示每个网格是否有效的
        """
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
