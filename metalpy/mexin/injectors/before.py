from .function_context import FunctionTermination
from .replaces import Replaces
from .utils import wrap_method_with_target


class Before(Replaces):
    def __init__(self, target, nest=None):
        super().__init__(target, keep_orig='__orig_fn__', nest=nest)

    def __call__(self, func):
        func, is_method = wrap_method_with_target(self.nest, func)

        def wrapper(*args, __orig_fn__=None, **kwargs):
            if is_method:
                _self = args[0]
                args = args[1:]
            ret = func(*args, **kwargs)
            if isinstance(ret, FunctionTermination):
                return ret.ret
            return __orig_fn__(*args, **kwargs)

        wrapper, is_method = wrap_method_with_target(self.nest, wrapper)
        return super().__call__(wrapper)


def before(target, nest=None):
    """将代码注入到目标函数前执行

    Parameters
    ----------
    target
        待注入的目标函数
    nest
        手动指定替换对象的所在空间

    Notes
    -----
        return terminate() 或 return terminate_with(*args) 可以指示提前终止函数

    See Also
    --------
        after: 将代码注入到函数之后执行
        replaces: 替换目标函数
        terminate or terminate_with: 指示以无返回值或指定返回值提前终止函数
    """
    return Before(target, nest=nest)
