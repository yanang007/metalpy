from typing import Union

from metalpy.utils.object_path import get_nest
from .recoverable_injector import RecoverableInjector
from .utils import wrap_method_with_target, create_replacement, get_ancestor


class Replaces(RecoverableInjector):
    def __init__(self, target, nest=None, keep_orig: Union[str, bool] = False, force_unbound=False):
        """替换目标函数

        Parameters
        ----------
        target
            待替换的目标，支持函数，方法，类名
        nest
            手动指定替换对象的所在空间，支持模块，类，实例
        keep_orig
            指示是否将原函数作为参数传入，当传入str时，将以该名字作为参数名
        force_unbound
            如果nest不为对象实例，该参数无效，如果nest为对象实例，则强制不对成员函数进行绑定

        Notes
        -----
            keep_orig=True时，原函数将以 **orig_fn** 的名字传入函数，此时请确保替换的函数包含这个名字的参数
            keep_orig为str时，原函数将以该名字传入函数，此时请确保替换的函数包含该指定名字的参数
        """
        super().__init__()
        target_name = target.__name__
        root_target = get_ancestor(target)
        if nest is None:
            nest = get_nest(root_target)
            if nest is None:
                raise NotImplementedError(
                    f'Unable locate where {root_target.__name__} is, consider specify it by "nest" param.'
                )

        self.nest = nest
        self.name = target_name
        self.backup = None
        self.keep_orig = keep_orig
        self.force_unbound = force_unbound

    def __call__(self, func):
        cmd = f'self.nest.{self.name}'
        self.backup = orig = eval(cmd)

        keep_orig = self.keep_orig
        if keep_orig and isinstance(keep_orig, bool):
            keep_orig = 'orig_fn'

        if keep_orig:
            orig_fn = {keep_orig: orig}
            wrapper = lambda *args, **kwargs: func(*args, **kwargs, **orig_fn)
        else:
            wrapper = lambda *args, **kwargs: func(*args, **kwargs)

        if not self.force_unbound:
            wrapper, is_method = wrap_method_with_target(self.nest, wrapper)
        wrapper = create_replacement(wrapper, orig, self)
        cmd = f'self.nest.{self.name} = wrapper'
        exec(cmd)

        return wrapper

    def rollback(self):
        cmd = f'self.nest.{self.name} = self.backup'
        exec(cmd)


def replaces(target, nest=None, keep_orig=False):
    """替换目标函数

    Parameters
    ----------
    target
        待替换的目标，支持函数，方法，类名
    nest
        手动指定替换对象的所在空间，支持模块，类，实例
    keep_orig
        指示是否将原函数作为参数传入，当传入str时，将以该名字作为参数名

    Notes
    -----
        keep_orig=True时，原函数将以 **orig_fn** 的名字传入函数，此时请确保替换的函数包含这个名字的参数
        keep_orig为str时，原函数将以该名字传入函数，此时请确保替换的函数包含该指定名字的参数
    """
    return Replaces(target, nest=nest, keep_orig=keep_orig)
