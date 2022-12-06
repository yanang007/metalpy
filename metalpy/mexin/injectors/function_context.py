from .recoverable_injector import RecoverableInjector
from .utils import wrap_method_with_target, create_replacement


class FunctionTermination:
    """指示提前终止函数，目前仅在before注解中使用，指示提前终止函数
    """
    def __init__(self, *rets):
        if len(rets) == 0:
            self.ret = None
        elif len(rets) == 1:
            self.ret = rets[0]
        else:
            self.ret = rets


def terminate():
    return FunctionTermination()


def terminate_with(*rets):
    return FunctionTermination(*rets)
