import numpy as np


class Bounds:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        """构造一个Bounds对象

        Parameters
        ----------
        xmin, xmax, ymin, ymax, zmin, zmax
            三方向边界值
        """
        self.bounds = np.array([xmin, xmax, ymin, ymax, zmin, zmax])

    @classmethod
    def of(cls, xmin, xmax, ymin, ymax, zmin, zmax):
        return cls(xmin, xmax, ymin, ymax, zmin, zmax)

    @staticmethod
    def merge(left, right):
        if isinstance(left, NoBounds):
            return right
        elif isinstance(right, NoBounds):
            return left
        else:
            bounds = np.zeros(6)
            bounds[1::2] = np.max([left[1::2], right[1::2]], axis=0)
            bounds[0::2] = np.min([left[0::2], right[0::2]], axis=0)
            return left.of(*bounds)

    def __getitem__(self, item):
        return self.bounds[item]

    @property
    def xmin(self):
        return self.bounds[0]

    @property
    def xmax(self):
        return self.bounds[1]

    @property
    def ymin(self):
        return self.bounds[2]

    @property
    def ymax(self):
        return self.bounds[3]

    @property
    def zmin(self):
        return self.bounds[4]

    @property
    def zmax(self):
        return self.bounds[5]

    @property
    def xspan(self):
        return self.xmax - self.xmin

    @property
    def yspan(self):
        return self.ymax - self.ymin

    @property
    def zspan(self):
        return self.zmax - self.zmin

    def __str__(self):
        return f'Bounds({self.bounds})'

    def __iter__(self):
        for e in self.bounds:
            yield e

    def __array__(self):
        return self.bounds


class NoBounds(Bounds):
    def __init__(self, xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0):
        """指示边界不参与计算

        Parameters
        ----------
        xmin, xmax, ymin, ymax, zmin, zmax
            占位符
        """
        super().__init__(0, 0, 0, 0, 0, 0)

    def __str__(self):
        return f'NoBounds()'
