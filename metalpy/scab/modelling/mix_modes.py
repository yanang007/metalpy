from enum import Enum


class MixMode(Enum):
    Normal = 'normal'  # 直接覆盖
    Max = 'max'  # 取最大值

    @staticmethod
    def dispatch(mode):
        if isinstance(mode, MixMode):
            if mode == MixMode.Normal:
                return MixMode.normal
            elif mode == MixMode.Max:
                return MixMode.max
        else:
            return None

    @staticmethod
    def normal(prev_layer, current_layer):
        return current_layer

    @staticmethod
    def max(prev_layer, current_layer):
        mask = prev_layer < current_layer
        prev_layer[mask] = current_layer[mask]
        return prev_layer
