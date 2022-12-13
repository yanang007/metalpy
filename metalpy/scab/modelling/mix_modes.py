from enum import Enum
from typing import Union, Callable

import numpy as np

from metalpy.utils.hash import hash_numpy_array, hash_string_value


class MixMode(Enum):
    @staticmethod
    def dispatch(mode):
        if isinstance(mode, MixMode):
            name = str(mode.value)
            return getattr(MixMode, name)
        elif callable(mode):
            return mode
        else:
            raise ValueError(f'Mix mode must be either MixMode or function, got {type(mode)} instead.')

    @staticmethod
    def override(prev_layer, current_layer):
        return current_layer

    @staticmethod
    def keep_original(prev_layer, current_layer):
        return prev_layer

    @staticmethod
    def max(prev_layer, current_layer):
        mask = prev_layer < current_layer
        prev_layer[mask] = current_layer[mask]
        return prev_layer

    @staticmethod
    def min(prev_layer, current_layer):
        mask = prev_layer > current_layer
        prev_layer[mask] = current_layer[mask]
        return prev_layer

    @staticmethod
    def mean(prev_layer, current_layer):
        return (prev_layer + current_layer) / 2

    Override = override.__func__.__name__  # 直接覆盖
    KeepOriginal = keep_original.__func__.__name__  # 保留原值
    Max = max.__func__.__name__  # 取最大值
    Min = min.__func__.__name__  # 取最小
    Mean = mean.__func__.__name__  # 取最小


Mixer = Union[MixMode, Callable]


def hash_mixer(mixer: Mixer):
    """返回mixer的哈希值

    Parameters
    ----------
    mixer
        混合器名或函数

    Warnings
    --------
        TODO: Python内置hash对字符串的哈希结果会随机化导致问题，需要更换为确定性哈希算法
    """
    if isinstance(mixer, MixMode):
        return hash_string_value(mixer.value)
    else:
        import cloudpickle
        buf = np.frombuffer(cloudpickle.dumps(mixer), dtype=np.uint8)
        return hash_numpy_array(buf)
