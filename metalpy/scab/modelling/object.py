from typing import Any, Union

from .mix_modes import MixMode, Mixer, dhashable_mixer
from .shapes import Shape3D
from ...utils.dhash import dhash


class Object:
    DEFAULT_KEY = '__active__'

    def __init__(self,
                 shape: Shape3D,
                 models: Union[dict[str, Any], Any],
                 mix_mode: Mixer = MixMode.Override):
        """代表一个三维几何体图层

        Parameters
        ----------
        shape
            三维几何体
        models
            三维几何体的参数
        mix_mode
            混合模式，必须为MixMode枚举或自定义的混合函数，函数的参数见 Object.mix
        """
        self.shape = shape
        if models is None:
            models = True
        if not isinstance(models, dict):
            models = {Object.DEFAULT_KEY: models}
        self.models = models

        self.mix_mode = mix_mode

    @property
    def mixer(self):
        return MixMode.dispatch(self.mix_mode)

    @property
    def shape(self) -> Shape3D:
        return self._shape

    @shape.setter
    def shape(self, val: Shape3D):
        self._shape = val

    @property
    def models(self) -> dict[str, Any]:
        return self._models

    @models.setter
    def models(self, val: dict[str, Any]):
        self._models = val

    def items(self):
        for k, v in self.models.items():
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
        return self.models[item]

    def __dhash__(self):
        return dhash(self.shape, dhashable_mixer(self.mix_mode), self.models)
