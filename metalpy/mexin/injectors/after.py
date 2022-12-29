from typing import Union

from .replaces import Replaces
from .utils import wrap_method_with_target


class After(Replaces):
    def __init__(self, target, nest=None, keep_retval: Union[bool, str] = False):
        super().__init__(target, nest=nest, keep_orig='__orig_fn__', force_unbound=True)
        self.keep_retval = keep_retval

    def __call__(self, func):
        func, is_method = wrap_method_with_target(self.nest, func)
        keep_retval = self.keep_retval
        if keep_retval and isinstance(keep_retval, bool):
            keep_retval = 'retval'

        def wrapper(*args, __orig_fn__=None, **kwargs):
            result = __orig_fn__(*args, **kwargs)

            if keep_retval:
                retval = {keep_retval: result}
                ret = func(*args, **kwargs, **retval)
            else:
                ret = func(*args, **kwargs)

            if ret is not None:
                result = ret
            return result

        return super().__call__(wrapper)


def after(target, nest=None, keep_retval=False):
    """将代码注入到目标函数返回后执行

    Parameters
    ----------
    target
        待注入的目标函数
    nest
        手动指定替换对象的所在空间
    keep_retval
        指示是否将原函数返回值作为参数传入

    Notes
    -----
        返回非空的值可以替换原函数的返回值
        keep_retval=True时，原函数返回值将以 **retval** 的名字传入函数，此时请确保替换的函数包含这个名字的参数
        keep_retval为str时，原函数返回值将以该名字传入函数，此时请确保替换的函数包含该指定名字的参数

    See Also
    --------
        before: 将代码注入到函数之前执行
        replaces: 替换目标函数
    """
    return After(target, nest=nest, keep_retval=keep_retval)
