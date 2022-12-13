from enum import Enum
from typing import Union, Callable


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
