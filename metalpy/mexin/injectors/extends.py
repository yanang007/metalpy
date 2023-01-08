import warnings

from metalpy.utils.object_path import DottedName, get_qualname, get_module_name
from .recoverable_injector import RecoverableInjector
from .utils import wrap_method_with_target
from .replacement import create_replacement


class Extends(RecoverableInjector):
    def __init__(self, nest, name):
        super().__init__()
        self.nest = nest
        self.name = name

    def __call__(self, func):
        if getattr(self.nest, self.name, None) is not None:
            warnings.warn('Trying to extends to a existing target, may lead to unexpected result.')
        wrapper, is_method = wrap_method_with_target(self.nest, func)

        nest_type = self.nest
        if is_method:
            nest_type = type(self.nest)

        wrapper = create_replacement(wrapper, None, self,
                                     qualname=DottedName(get_qualname(nest_type), self.name),
                                     module_name=get_module_name(nest_type))

        cmd = f'self.nest.{self.name} = wrapper'
        exec(cmd)

        return wrapper

    def rollback(self):
        cmd = f'del self.nest.{self.name}'
        exec(cmd)


def extends(target, name):
    return Extends(target, name)
