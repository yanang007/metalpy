from typing import Any, Union, Callable

from .mix_modes import MixMode
from .shapes import Shape3D


class Object:
    DEFAULT_KEY = '__active__'

    def __init__(self,
                 shape: Shape3D,
                 values: Union[dict[str, Any], Any],
                 mix_mode: Union[MixMode, Callable] = MixMode.Normal):
        """代表一个三维几何体图层

        Parameters
        ----------
        shape
            三维几何体
        values
            三维几何体的参数
        mix_mode
            混合模式，必须为MixMode枚举或自定义的混合函数，函数的参数见 Object.mix
        """
        self.shape = shape
        if values is None:
            values = True
        if not isinstance(values, dict):
            values = {Object.DEFAULT_KEY: values}
        self.values = values

        if isinstance(mix_mode, MixMode):
            self.mixer = MixMode.dispatch(mix_mode)
        elif callable(mix_mode):
            self.mixer = mix_mode
        else:
            raise ValueError(f'Mix mode must be either MixMode or function, got {type(mix_mode)} instead.')

    @property
    def shape(self) -> Shape3D:
        return self._shape

    @shape.setter
    def shape(self, val: Shape3D):
        self._shape = val

    @property
    def values(self) -> dict[str, Any]:
        return self._values

    @values.setter
    def values(self, val: dict[str, Any]):
        self._values = val

    def items(self):
        for k, v in self.values.items():
            yield k, v

    def mix(self, prev_layer, current_layer):
        """执行模型混合操作

        Parameters
        ----------
        prev_layer
            之前所有模型结果和当前模型的重合部分的值数组
        current_layer
            当前模型重合部分的值数据

        Returns
        -------
            混合操作后的结果
        """
        return self.mixer(prev_layer, current_layer)

    def __getitem__(self, item):
        return self.values[item]
