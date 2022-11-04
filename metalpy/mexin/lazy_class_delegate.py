class LazyClassFactory:
    """用于劫持类的构造函数，实现延迟构造，常配合.injectors.hijack使用
    通过construct可以使用对应的类进行构造，new_kwargs指定用于替换或追加参数
    通过clone可以复制一个新的延迟构造器
    """
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = list(args)  # 方便删除
        self.kwargs = kwargs

    def find_param(self, predicate, remove=False):
        ret = None
        for i, arg in enumerate(self.args):
            if predicate(i, arg):
                ret = arg
                if remove:
                    del self.args[i]
                break

        if ret is None:
            for key, kwarg in self.kwargs.items():
                if predicate(key, kwarg):
                    ret = kwarg
                    if remove:
                        del self.kwargs[key]
                    break

        return ret

    def find_param_by_type(self, type, remove=False):
        return self.find_param(lambda k, v: isinstance(v, type), remove=remove)

    def find_param_by_name(self, name, remove=False):
        return self.find_param(lambda k, v: k == name, remove=remove)

    def construct(self, **new_kwargs):
        kwargs = self.kwargs.copy()
        kwargs.update(new_kwargs)

        checked_args = (arg.construct() if isinstance(arg, LazyClassFactory) else arg for arg in self.args)
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
