from typing import Any


class LazyClassFactory:
    """用于劫持类的构造函数，实现延迟构造，常配合.injectors.hijack使用
    通过construct可以使用对应的类进行构造，new_kwargs指定用于替换或追加参数
    通过clone可以复制一个新的延迟构造器
    """
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = list(args)  # 方便删除
        self.kwargs = kwargs

    def find_param(self, predicate, remove=False, repl=None, args=None, kwargs=None):
        if args is None:
            args = self.args
        if kwargs is None:
            kwargs = self.kwargs

        ret = None
        if args != False:
            for i, arg in enumerate(args):
                if predicate(i, arg):
                    ret = arg
                    if repl is not None:
                        args[i] = repl
                    elif remove:
                        del args[i]
                    break

        if ret is None and kwargs != False:
            for key, kwarg in kwargs.items():
                if predicate(key, kwarg):
                    ret = kwarg
                    if repl is not None:
                        kwargs[key] = repl
                    elif remove:
                        del kwargs[key]
                    break

        return ret

    def find_param_by_type(self, type, remove=False, repl=None, args=None, kwargs=None):
        return self.find_param(lambda k, v: isinstance(v, type), remove=remove, args=args, kwargs=kwargs)

    def find_param_by_name(self, name, remove=False, repl=None, args=None, kwargs=None):
        return self.find_param(lambda k, v: k == name, remove=remove, args=args, kwargs=kwargs)

    def construct(self, replace_by_type: dict[type, Any] = None, **new_kwargs):
        args = self.args.copy()
        kwargs = self.kwargs.copy()

        if replace_by_type is not None:
            for k in replace_by_type:
                if isinstance(k, type):
                    self.find_param_by_type(args, repl=replace_by_type[k])

        kwargs.update(new_kwargs)

        checked_args = (arg.construct() if isinstance(arg, LazyClassFactory) else arg for arg in args)
        checked_kwargs = kwargs
        for key in checked_kwargs:
            value = checked_kwargs[key]
            if isinstance(value, LazyClassFactory):
                checked_kwargs[key] = value.construct()

        return self.cls(*checked_args, **checked_kwargs)

    def clone(self, **new_kwargs):
        kwargs = self.kwargs.copy()
        kwargs.update(new_kwargs)
        return LazyClassFactory(self.cls, *self.args, **kwargs)

    def __getitem__(self, key):
        return self.kwargs[key]

    def __setitem__(self, key, value):
        self.kwargs[key] = value

    def __delitem__(self, key):
        if key in self.kwargs:
            del self.kwargs[key]
        else:
            del self.args[key]
