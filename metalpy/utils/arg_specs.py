from __future__ import annotations

import inspect
import warnings
from typing import Callable, Any

from metalpy.utils.type import undefined


class Arg:
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Arg):
            return self.name == other.name
        return False

    def __repr__(self):
        return f'{Arg.__name__}("{self.name}")'

    def __str__(self):
        return Arg.__name__


class ArgSpecKey:
    def __init__(self, key):
        self.key = key


class ArgSpecs:
    def __init__(self):
        self.posarg_specs: list[Arg] = []
        self.kwarg_specs = set()
        self.n_required_posargs = 0
        self.required_kwargs = set()

        self.contiguous_posargs_count = 0
        self.bound_posargs: dict[int, Any] = {}
        self.bound_kwargs: dict[Arg, Any] = {}
        self.bound_varargs: list[Any] | None = None
        self.bound_varkw: dict[Arg, Any] | None = None

        self.immutable = False

    @staticmethod
    def of(fn: Callable):
        ret = ArgSpecs()

        spec = inspect.getfullargspec(fn)

        args = spec.args
        if isinstance(fn, type):
            args = args[1:]

        kwonlyargs = spec.kwonlyargs
        defaults = spec.defaults or []
        kwonlydefaults = spec.kwonlydefaults or {}

        if spec.varargs is not None:
            ret.accepts_varargs = True

        if spec.varkw is not None:
            ret.accepts_varkw = True

        n_required_posargs = len(args) - len(defaults)
        for i, arg_name in enumerate(args):
            ret.append_arg_spec(arg_name, required=i < n_required_posargs, kwonly=False)

        for arg_name in kwonlyargs:
            ret.append_arg_spec(arg_name, required=arg_name not in kwonlydefaults, kwonly=True)

        return ret

    @staticmethod
    def from_class_mro(cls):
        ret = ArgSpecs.of(cls)

        for parent in inspect.getmro(cls)[1:]:
            ret.merge_kwargs(parent)

        return ret

    def merge_kwargs(self, fn):
        """将可调用对象fn的所有允许使用关键字调用的参数合并到当前ArgSpecs实例的kwargs中

        Parameters
        ----------
        fn
            待分析的fn

        Returns
        -------
        self
            更新后的ArgSpecs实例

        Notes
        -----
            仅在当前参数定义接收kwargs有效，否则会报错
        """
        self._check_immutable()

        if not self.accepts_varkw:
            raise ValueError('Merging kwargs into a function without kwargs is semantically forbidden.')

        spec = inspect.getfullargspec(fn)

        args = spec.args
        if isinstance(fn, type):
            args = args[1:]

        kwonlyargs = spec.kwonlyargs
        defaults = spec.defaults or []
        kwonlydefaults = spec.kwonlydefaults or {}

        n_required_posargs = len(args) - len(defaults)
        for i, arg_name in enumerate(args):
            if self.find_arg_key(arg_name) is None:
                self.append_arg_spec(arg_name, required=i < n_required_posargs, kwonly=True)

        for arg_name in kwonlyargs:
            if self.find_arg_key(arg_name) is None:
                self.append_arg_spec(arg_name, required=arg_name not in kwonlydefaults, kwonly=True)

        return self

    @property
    def accepts_varargs(self):
        return self.bound_varargs is not None

    @accepts_varargs.setter
    def accepts_varargs(self, b: bool):
        if b:
            self.bound_varargs = []
        else:
            if self.bound_varargs is not None and len(self.bound_varargs) > 0:
                warnings.warn('Disabling `varargs` with existing bound values.')
            self.bound_varargs = None

    @property
    def accepts_varkw(self):
        return self.bound_varkw is not None

    @accepts_varkw.setter
    def accepts_varkw(self, b: bool):
        if b:
            self.bound_varkw = {}
        else:
            if self.bound_varkw is not None and len(self.bound_varkw) > 0:
                warnings.warn('Disabling `varkw` with existing bound values.')
            self.bound_varkw = None

    @property
    def n_posargs(self):
        return len(self.posarg_specs)

    @property
    def n_kwargs(self):
        return len(self.kwarg_specs)

    @property
    def n_required_kwargs(self):
        return len(self.required_kwargs)

    @property
    def n_required(self):
        return self.n_required_posargs + self.n_required_kwargs

    @property
    def n_args(self):
        return self.n_posargs + self.n_kwargs

    @property
    def n_varargs(self):
        return 0 if not self.accepts_varargs else len(self.bound_varargs)

    @property
    def n_varkw(self):
        return 0 if not self.accepts_varkw else len(self.bound_varkw)

    @property
    def is_posargs_contiguous(self):
        """指示当前绑定的位置参数是否连续（即每个位置参数之前的所有参数都已给定）
        """
        return self.contiguous_posargs_count == len(self.bound_posargs)

    @property
    def is_satisfied(self):
        return self.is_posargs_satisfied and self.is_kwargs_satisfied

    @property
    def is_posargs_satisfied(self):
        return self.contiguous_posargs_count > self.n_required_posargs

    @property
    def is_all_posargs_provided(self):
        return self.contiguous_posargs_count == self.n_posargs

    @property
    def is_kwargs_satisfied(self):
        return len(self.missing_kwargs) == 0

    @property
    def missing_posargs(self):
        return self.posarg_specs[self.contiguous_posargs_count:self.n_required_posargs]

    @property
    def missing_kwargs(self):
        return self.required_kwargs.difference(set(self.bound_kwargs.keys()))

    def __contains__(self, item):
        ret = self.find_arg_key(item)
        if ret is not None:
            return True
        else:
            return False

    def append_arg_spec(self, name, kwonly=False, required=True):
        self._check_immutable()
        arg = Arg(name)
        if kwonly:
            if required:
                self.required_kwargs.add(arg)
            self.kwarg_specs.add(arg)
        else:
            if required:
                self.n_required_posargs += 1
            self.posarg_specs.append(arg)

    def push_args(self, *args):
        self._mark_immutable()

        if not self.is_posargs_contiguous:
            raise RuntimeError('Pushing args to noncontiguous posargs is not allowed.')

        n_seq_args = self.contiguous_posargs_count
        n_new_args = len(args)

        if not self.accepts_varargs and n_seq_args + n_new_args > self.n_posargs:
            raise ValueError(f'Pushing too many values to fixed-length posargs[{self.n_posargs}]'
                             f' (`accepts_varargs` == False).')

        n_non_varargs = max(n_seq_args + n_new_args - self.n_posargs, 0)
        for i, arg in zip(
            range(n_seq_args, n_seq_args + n_non_varargs),
            args[:n_non_varargs]
        ):
            self._bind_arg_by_index_unsafe(i, arg)

        self.contiguous_posargs_count = n_seq_args + n_non_varargs

        if n_non_varargs < n_new_args:
            self._extend_varargs(args[n_non_varargs:])

    def bind_kwargs(self, **kwargs):
        # self._mark_immutable()

        for k, v in kwargs.items():
            self.bind_arg(k, v)

    def bind_arg(self, name_or_index: str | int | ArgSpecKey, value):
        self._mark_immutable()

        name_or_index = self.find_arg_key(name_or_index, raises=True)
        self._bind_or_get_value_unsafe(name_or_index, value)

    def get_bound_arg(self, name_or_index: str | int | ArgSpecKey, default=undefined):
        name_or_index = self.find_arg_key(
            name_or_index,
            raises=default == undefined,
            default=default
        )
        return self._bind_or_get_value_unsafe(name_or_index)

    def call(self, func):
        posargs, kwargs = self.build_all_args()
        return func(*posargs, **kwargs)

    def build_all_args(self):
        posargs = []

        missing_posargs = self.missing_posargs
        missing_kwargs = list(self.missing_kwargs)
        missing_tips = []

        if len(missing_posargs) > 0:
            missing_tips.append(f'posarg(s) `{"`, `".join([m.name for m in missing_posargs])}`')

        if len(missing_kwargs) > 0:
            missing_tips.append(f'kwarg(s) `{"`, `".join([m.name for m in missing_kwargs])}`')

        if len(missing_posargs) + len(missing_kwargs) > 0:
            missing_str = f'Missing required {" and ".join(missing_tips)}.'
            raise ValueError(missing_str, missing_posargs, missing_kwargs)

        for i in range(self.contiguous_posargs_count):
            posargs.append(self.bound_posargs[i])

        if self.accepts_varargs and self.n_varargs > 0:
            if not self.is_all_posargs_provided:
                raise ValueError('Cannot bind varargs when posargs are not all specified.',
                                 self.posarg_specs[self.contiguous_posargs_count:self.n_posargs],
                                 [])
            else:
                posargs += self.bound_varargs

        kwargs = {k.name: v for k, v in self.bound_kwargs.items()}

        for i in range(self.contiguous_posargs_count, self.n_posargs):
            if i in self.bound_posargs:
                kwargs[self.posarg_specs[i].name] = self.bound_posargs[i]

        if self.accepts_varkw:
            for k, v in self.bound_varkw.items():
                kwargs[k.name] = v

        return posargs, kwargs

    def find_arg_key(self, name_or_index, default=None, raises=False):
        if isinstance(name_or_index, ArgSpecKey):
            return name_or_index
        elif isinstance(name_or_index, int):
            if not self.accepts_varargs and self.n_posargs < name_or_index:
                raise KeyError(f'Index `{name_or_index}` out of range of fixed-length posargs[{self.n_posargs}]'
                               f' (`accepts_varargs` == False).')
            return ArgSpecKey(name_or_index)
        elif isinstance(name_or_index, str):
            arg = Arg(name_or_index)
            if arg in self.kwarg_specs:
                return ArgSpecKey(name_or_index)
            else:
                try:
                    index = self.posarg_specs.index(arg)
                    return ArgSpecKey(index)
                except ValueError:
                    pass

        if raises:
            raise KeyError(f'`{name_or_index}` does not match any of arg specs.')
        else:
            return default

    def clear(self):
        self._mark_immutable(False)

    def clone_specs(self):
        """只复制参数定义本身，不复制参数的绑定值

        Returns
        -------
        ret
            新的实例
        """
        ret = ArgSpecs()

        ret.posarg_specs = self.posarg_specs
        ret.kwarg_specs = self.kwarg_specs
        ret.n_required_posargs = self.n_required_posargs
        ret.required_kwargs = self.required_kwargs

        ret.contiguous_posargs_count = self.contiguous_posargs_count
        ret.bound_posargs = self.bound_posargs
        ret.bound_kwargs = self.bound_kwargs
        ret.bound_varargs = self.bound_varargs
        ret.bound_varkw = self.bound_varkw

        return ret

    def _check_contiguous(self):
        while self.contiguous_posargs_count in self.bound_posargs:
            self.contiguous_posargs_count += 1

    def _extend_varargs(self, new_varargs):
        self.bound_varargs.extend(new_varargs)

    def _bind_arg_by_index_unsafe(self, index, val=undefined):
        if index < self.n_posargs:
            if val == undefined:
                return self.bound_posargs[index]
            else:
                self.bound_posargs[index] = val
        else:
            if index - self.n_posargs != self.n_varargs:
                warnings.warn('Varargs do not support binding uncontiguously by index. The value will be appended.')

            if val == undefined:
                return self.bound_varargs[index]
            else:
                self.bound_varargs.append(val)

    def _bind_kwarg_by_name_unsafe(self, name, val=undefined):
        name = Arg(name)
        if name in self.kwarg_specs:
            if undefined == val:
                return self.bound_kwargs[name]
            else:
                self.bound_kwargs[name] = val
        elif self.accepts_varkw:
            if undefined == val:
                return self.bound_varkw[name]
            else:
                self.bound_varkw[name] = val

    def _bind_or_get_value_unsafe(self, key, value=undefined):
        if isinstance(key, ArgSpecKey):
            key = key.key

        if isinstance(key, str):
            if undefined == value:
                return self._bind_kwarg_by_name_unsafe(key)
            else:
                self._bind_kwarg_by_name_unsafe(key, value)

        if isinstance(key, int):
            if undefined == value:
                return self._bind_arg_by_index_unsafe(key)
            else:
                self._bind_arg_by_index_unsafe(key, value)
                self._check_contiguous()

    def _mark_immutable(self, b=True):
        """指示该ArgSpecs实例已经开始绑定值，不再允许修改参数定义
        """
        self.immutable = b

    def _check_immutable(self):
        """检查该ArgSpecs实例是否允许修改参数定义
        """
        if self.immutable:
            raise RuntimeError(f'Modifying arg spec when there are bound values is not allowed.')

    def __repr__(self):
        args = []
        for i in range(self.n_required_posargs):
            args.append(self.posarg_specs[i].name)

        for i in range(self.n_required_posargs, self.n_posargs):
            args.append(f"[{self.posarg_specs[i].name}]")

        if self.accepts_varargs:
            args.append("*args")
        else:
            args.append("*")

        for arg in self.required_kwargs:
            args.append(arg.name)

        for arg in self.kwarg_specs.difference(self.required_kwargs):
            args.append(f'[{arg.name}]')

        if args[-1] == '*':
            # 即没有仅关键字参数
            args.pop()

        if self.accepts_varkw:
            args.append("**kwargs")

        return f'ArgSpecs({", ".join(args)})'
