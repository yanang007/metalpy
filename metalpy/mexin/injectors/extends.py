from .recoverable_injector import RecoverableInjector
from .utils import wrap_method_with_target, create_replacement


class Extends(RecoverableInjector):
    def __init__(self, nest, name):
        super().__init__()
        self.nest = nest
        self.name = name

    def __call__(self, func):
        wrapper = wrap_method_with_target(self.nest, func)
        wrapper = create_replacement(wrapper, None, self, name=self.name)
        cmd = f'self.nest.{self.name} = wrapper'
        exec(cmd)
        return wrapper

    def rollback(self):
        cmd = f'del self.nest.{self.name}'
        exec(cmd)


def extends(target, name):
    return Extends(target, name)
