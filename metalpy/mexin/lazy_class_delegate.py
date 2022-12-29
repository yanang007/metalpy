import inspect
from typing import Any


class LazyClassFactory:
    def __init__(self, cls, *args, **kwargs):
        """用于实现类的延迟构造，
        通过construct可以使用对应的类进行构造，new_kwargs指定用于替换或追加的参数，
        通过clone可以复制一个新的延迟构造器
        
        Parameters
        ----------
        cls
            待构造目标类
        args
            位置参数
        kwargs
            关键字参数
        """
        self.cls = cls
        self.args = list(args)  # 方便删除
        self.kwargs = kwargs

    def find_param(self, predicate, remove=False, repl=None, args=None, kwargs=None):
        """在参数列表中搜索指定参数

        Parameters
        ----------
        predicate
            搜索判断条件
        remove
            指示是否移除搜索到的参数
        repl
            替换项，如果搜索到结果且repl不为None则用该参数替换
        args
            待搜索的位置参数，为None则使用self的args，为False则忽略args
        kwargs
            待搜索的关键字参数，为None则使用self的kwargs，为False则忽略kwargs

        Returns
        -------
            搜索到的目标参数
        """
        if args is None:
            args = self.args
        if kwargs is None:
            kwargs = self.kwargs

        arg_spec = inspect.getfullargspec(self.cls)
        args_info = arg_spec.args[1:]  # 构造函数必定以self作为第一个参数，排除掉
        ret = None
        if args != False:
            for i, arg in enumerate(args):
                matched = predicate(i, arg)
                if not matched and i < len(args_info):
                    arg_name = args_info[i]
                    matched = matched or predicate(arg_name, arg)
                if matched:
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
        """在参数列表中搜索指定类型的参数

        Parameters
        ----------
        type
            指定的参数类型
        remove
            指示是否移除搜索到的参数
        repl
            替换项，如果搜索到结果且repl不为None则用该参数替换
        args
            待搜索的位置参数，为None则使用self的args，为False则忽略args
        kwargs
            待搜索的关键字参数，为None则使用self的kwargs，为False则忽略kwargs

        Returns
        -------
            搜索到的目标参数
        """
        return self.find_param(lambda k, v: isinstance(v, type), remove=remove, repl=repl, args=args, kwargs=kwargs)

    def find_param_by_name(self, name, remove=False, repl=None, args=None, kwargs=None):
        """在参数列表中搜索指定名字的参数

        Parameters
        ----------
        name
            指定的参数名
        remove
            指示是否移除搜索到的参数
        repl
            替换项，如果搜索到结果且repl不为None则用该参数替换
        args
            待搜索的位置参数，为None则使用self的args，为False则忽略args
        kwargs
            待搜索的关键字参数，为None则使用self的kwargs，为False则忽略kwargs

        Returns
        -------
            搜索到的目标参数
        """
        return self.find_param(lambda k, v: k == name, remove=remove, repl=repl, args=args, kwargs=kwargs)

    def construct(self, replace_by_type: dict[type, Any] = None, **new_kwargs):
        """构造类对象

        Parameters
        ----------
        replace_by_type
            需要按类型替换的参数列表
        new_kwargs
            新的关键字参数

        Returns
        -------
            构造的类对象
        """
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
