import inspect
import types

from metalpy.mexin.injectors.replacement import get_orig


def wrap_method_with_target(target, func):
    if target is not None and not isinstance(target, (type, type(types))):
        # 目标不是None、类型与模块，则是实例，func是实例方法，直接绑定
        wrapper = types.MethodType(func, target)
        is_target_method = True
    else:
        wrapper = func
        is_target_method = False

    return wrapper, is_target_method


def update_params(func, args, kwargs, new_kwargs):
    """使用new_kwargs更新func的args和kwargs

    Parameters
    ----------
    func
        目标函数签名
    args
        目标函数原有的位置参数
    kwargs
        目标函数原有的关键字参数
    new_kwargs
        新的参数列表，使用str修改关键字或非varargs的位置参数，使用int修改varargs中指定位置的参数

    Returns
    -------
        tuple(updated_args, updated_kwargs)
        更新后的args和kwargs
    """
    from metalpy.utils.type import get_or_default, not_none_or_default

    # func可能被替换过，导致函数签名和原函数不一致，因此尝试回溯原函数从而获取真实函数签名
    # get_orig在不存在原函数时会返回None，因此需要判断
    func = not_none_or_default(get_orig(func), _default=func)
    arg_spec = inspect.getfullargspec(func)
    updated_args = None
    updated_kwargs = None

    arg_spec_args = arg_spec.args
    if isinstance(func, types.MethodType):
        # func是实例方法，需要排除掉self参数
        arg_spec_args = arg_spec_args[1:]

    for i, arg_name in enumerate(arg_spec_args):
        new_arg = get_or_default(new_kwargs, arg_name, _default=None)
        if new_arg is not None:
            if i < len(args):
                # 如果下标在args范围内，则一定是对应位置的参数
                if updated_args is None:
                    updated_args = list(args)
                updated_args[i] = new_arg
            else:
                # 如果下标不在args范围内，则原参数没有被指定或在kwargs中指定，无论哪种情况都应放到kwargs中
                if updated_kwargs is None:
                    updated_kwargs = dict(kwargs)
                updated_kwargs[arg_name] = new_arg

    for i in range(len(args)):
        new_arg = get_or_default(new_kwargs, i, _default=None)
        if new_arg is not None:
            # 通过下标替换args中的参数，主要用于varargs参数
            if updated_args is None:
                updated_args = list(args)
            updated_args[i] = new_arg

    for arg_name in arg_spec.kwonlyargs:
        new_arg = get_or_default(new_kwargs, arg_name, _default=None)
        if new_arg is not None:
            # 通过关键字替换kwargs中的kwonlyargs
            if updated_kwargs is None:
                updated_kwargs = dict(kwargs)
            updated_kwargs[arg_name] = new_arg

    if updated_args is None:
        updated_args = args
    if updated_kwargs is None:
        updated_kwargs = kwargs

    return updated_args, updated_kwargs
