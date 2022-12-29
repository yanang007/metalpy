import copy


class FunctionTermination:
    def __init__(self, *rets):
        """指示提前终止函数，目前仅在before注解中使用，指示提前终止函数
        """
        if len(rets) == 0:
            self.ret = None
        elif len(rets) == 1:
            self.ret = rets[0]
        else:
            self.ret = rets


def terminate():
    """指示无返回值提前终止函数执行

    Returns
    -------
        FunctionTermination，由Before注解接收并执行函数终止操作
    """
    return FunctionTermination()


def terminate_with(*rets):
    """指示以指定的返回值提前终止函数

    Parameters
    ----------
    rets
        指定函数终止的返回值

    Returns
    -------
        FunctionTermination，由Before注解接收并执行函数终止操作，然后返回指定的返回值
    """
    return FunctionTermination(*rets)


class ParameterModification:
    def __init__(self, new_args):
        """指示修改函数参数，目前仅在before注解中使用，指示修改函数参数
        """
        self.new_args = new_args


def modify_params(new_args=None, **new_kwargs):
    """指示修改函数参数

    Parameters
    ----------
    new_args
        以字典形式传入的参数修改指示，int作为key用于修改指定位置的位置参数，str作为key用于修改关键字参数或以关键字形式指定的位置参数
    new_kwargs
        以关键字形式传入的便捷参数修改指示，用于修改关键字参数或以关键字形式指定的位置参数

    Returns
    -------
        ParameterModification，由Before注解接收并执行参数修改
    """
    new_args = copy.copy(new_args)
    new_args.update(new_kwargs)
    return ParameterModification(new_args)
