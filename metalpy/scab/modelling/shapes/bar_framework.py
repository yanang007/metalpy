from itertools import product
from typing import Iterable

import numpy as np

from . import Shape3D, Cuboid
from .composition import Composition
from ..mix_modes import MixMode


def clarify_spec(spec, n_params=3):
    """定义一个长度为n_params的数组

    如果spec为可迭代类型，则验证spec长度是否满足。

    如果spec不为可迭代类型，则认为用单个spec定义了数组中的所有值，返回[spec] * n_params

    Parameters
    ----------
    spec
        待验证的参数
    n_params
        期望的参数个数

    Returns
    -------
    ret
        长度为n_params的数组，其值由spec定义
    """
    if isinstance(spec, Iterable):
        spec = list(spec)
        assert len(spec) == n_params
    else:
        spec = [spec] * n_params

    return np.asarray(spec)


class BarFramework(Composition):
    def __init__(self, outline: Shape3D, bar_spec, n_rooms=None, room_spec=None):
        """
        Parameters
        ----------
        outline
            边界，一般为Cuboid或者立方体Prism
        bar_spec
            三个方向上条柱的粗细，传入单一值时代表同时定义三个方向
        n_rooms
            三个方向上的房间数，传入单一值时代表同时定义三个方向
        room_spec
            三个方向上的房间尺寸，传入单一值时代表同时定义三个方向。
            房间尺寸以条柱中心为基准计算，保证结果中的房间尺寸为最大的不大于room_spec的等距划分。
            例如x方向有效宽度为12m，约定房间尺寸为11m，则结果中x方向房间数为2，单个房间尺寸为6m。
        """
        self.bar_spec = bar_spec = clarify_spec(bar_spec)

        if n_rooms is None:
            # 假设使用房间尺寸定义
            # 则保证生成的房间不大于指定的尺寸
            # ni = ceil(bi / ri)
            room_spec = clarify_spec(room_spec)
            # 约定条柱不会超出outline定义的边界
            # 因此从边界条柱的中心开始算，实际的有效空间大小有缩小
            sizes = outline.local_bounding_size - bar_spec
            n_rooms = np.ceil(sizes / room_spec).astype(int)
        else:
            n_rooms = clarify_spec(n_rooms)

        self._n_rooms = n_rooms

        self.outline = outline
        bars = [*self._create_bars(0), *self._create_bars(1), *self._create_bars(2)]
        super().__init__(
            outline,
            Composition(*bars, mix_mode=MixMode.Max),
            mix_mode=MixMode.Min
        )

    def bar_radius(self, axis):
        return self.bar_spec[axis] / 2

    def n_rooms(self, axis):
        return self._n_rooms[axis]

    def axis_span(self, axis):
        return self.outline.local_bounds[axis * 2], self.outline.local_bounds[axis * 2 + 1]

    def centers(self, axis):
        a0, a1 = self.axis_span(axis)
        ra = self.bar_radius(axis)
        na = self.n_rooms(axis)
        return np.linspace(a0 + ra, a1 - ra, na)

    def _create_bars(self, axis):
        # 长轴为c，坐标为a，b
        a, b, c = axes = np.r_[axis + 1, axis + 2, axis] % 3  # 三个轴的下标
        ra, rb = self.bar_radius(a), self.bar_radius(b)
        c0, c1 = self.axis_span(c)

        for ax, bx in product(self.centers(a), self.centers(b)):
            origin = np.r_[ax - ra, bx - rb, c0][axes]
            end = np.r_[ax + ra, bx + rb, c1][axes]
            yield Cuboid(corner=origin, corner2=end)

    @property
    def bars(self):
        return self[1]

    def to_local_polydata(self):
        return self.bars.to_local_polydata()
