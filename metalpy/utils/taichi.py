import copy
import functools
import os
import warnings
from typing import Sized

import taichi as ti
from taichi import TaichiSyntaxError, TaichiCompilationError, TaichiRuntimeError
from taichi.lang.enums import AutodiffMode
from taichi.lang.kernel_impl import _inside_class, Kernel
from taichi.lang.util import to_taichi_type

from .file import make_cache_directory, make_cache_file

ti_cache_prefix = 'taichi_cache'

ti_default_args = {
    'arch': ti.cpu,
    'offline_cache': True,
    'offline_cache_file_path': make_cache_directory(ti_cache_prefix),
}
ti_args = copy.deepcopy(ti_default_args)
ti_inited = False


def ti_prepare(**kwargs):
    ti_args.update(kwargs)


def ti_init_once():
    global ti_inited
    if not ti_inited:
        ti.init(**ti_args)
        ti_inited = True


def ti_reset():
    """重置taichi的配置参数，使得下次调用涉及kernel的函数时会重新初始化

    同时也用于解决taichi目前没有gc导致内存泄露的问题
    taichi-dev/taichi#6803: Improve garbage collection
    https://github.com/taichi-dev/taichi/issues/6803

    Notes
    -----
        reset会在下次init时生效，因此不会立刻解决内存泄露的问题
    """
    global ti_inited, ti_args
    ti_args = copy.deepcopy(ti_default_args)
    ti_inited = False


class WrappedTaichiContext:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        ti_prepare(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ti_reset()


def ti_config(**kwargs):
    return WrappedTaichiContext(**kwargs)


def ti_arch(arch):
    arch_type = type(ti.cpu)
    if isinstance(arch, str):
        return getattr(ti, arch)
    elif isinstance(arch, arch_type):
        return arch
    elif isinstance(arch, list) and isinstance(arch[0], arch_type):
        return arch
    else:
        raise ValueError(f"Unknown architecture, "
                         f"should be one of 'cpu', 'gpu' or specific backends like 'cuda' or 'vulkan'. ")


def ti_func(fn, is_real_function=None):
    if is_real_function is None:
        return ti.func(fn)
    else:
        return ti.func(fn, is_real_function=is_real_function)


def ti_real_func(fn):
    return ti_func(fn, is_real_function=True)


def ti_kernel(fn):
    # TODO: 由于ti.kernel使用了一个hack来判断是否定义在class内，
    #  因此这里改造了一份ti.lang.kernel_impl._kernel_impl来实现自动初始化
    # Can decorators determine if a function is being defined inside a class?
    # https://stackoverflow.com/a/8793684/12003165
    is_classkernel = _inside_class(2 + 1)

    primal = Kernel(fn,
                    autodiff_mode=AutodiffMode.NONE,
                    _classkernel=is_classkernel)
    adjoint = Kernel(fn,
                     autodiff_mode=AutodiffMode.REVERSE,
                     _classkernel=is_classkernel)
    # Having |primal| contains |grad| makes the tape work.
    primal.grad = adjoint

    if is_classkernel:
        # For class kernels, their primal/adjoint callables are constructed
        # when the kernel is accessed via the instance inside
        # _BoundedDifferentiableMethod.
        # This is because we need to bind the kernel or |grad| to the instance
        # owning the kernel, which is not known until the kernel is accessed.
        #
        # See also: _BoundedDifferentiableMethod, data_oriented.
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # If we reach here (we should never), it means the class is not decorated
            # with @ti.data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            assert not hasattr(clsobj, '_data_oriented')
            raise TaichiSyntaxError(
                f'Please decorate class {clsobj.__name__} with @ti.data_oriented'
            )
    else:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                ti_init_once()  # 完成初始化
                return primal(*args, **kwargs)
            except (TaichiCompilationError, TaichiRuntimeError) as e:
                raise type(e)('\n' + str(e)) from None

        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    return wrapped


def ti_data_oriented(cls):
    return ti.data_oriented(cls)


def ti_ndarray(dtype, shape):
    ti_init_once()
    return ti.ndarray(dtype, shape)


def ti_root():
    ti_init_once()
    return ti.root


def ti_cfg():
    ti_init_once()
    return ti.cfg


def ti_field(dtype,
             shape=None,
             order=None,
             name="",
             offset=None,
             needs_grad=False,
             needs_dual=False):
    ti_init_once()
    return ti.field(dtype, shape, order, name, offset, needs_grad, needs_dual)


class WrappedFieldsBuilder:
    def __init__(self):
        self.fields_builder = ti.FieldsBuilder()
        self.snode_tree = None

    def finalize(self, raise_warning=True):
        self.snode_tree = self.fields_builder.finalize(raise_warning)
        return self.snode_tree

    def destroy(self):
        if self.snode_tree is not None:
            self.snode_tree.destroy()

    def place_dense(self, shape, *arr_types, axes=None):
        if axes is None:
            if isinstance(shape, Sized):
                ndim = len(shape)
            else:
                ndim = 1
            axes = ti.axes(*list(range(ndim)))
        arr_types = tuple(wrap_type_as_ti_field(t) for t in arr_types)
        self.dense(axes, shape).place(*arr_types)

        if len(arr_types) == 1:
            return arr_types[0]
        else:
            return arr_types

    def __getattr__(self, name):
        return getattr(self.fields_builder, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


def wrap_type_as_ti_field(t):
    if not isinstance(t, ti.Field):
        return ti_field(t)
    return t


def ti_FieldsBuilder():
    ti_init_once()
    return WrappedFieldsBuilder()


def ti_test_snode_support() -> bool:
    """测试cuda版本是否支持SNode和FieldsBuilder

    如果不支持，则需要使用动态数组ti.ndarray来代替

    Returns
    -------
    ret
        当前显卡cuda版本是否支持SNode和FieldsBuilder
    """
    ti_init_once()
    snode_support_indicator_cache = make_cache_file(f'{ti_cache_prefix}/snode_support')
    if os.path.exists(snode_support_indicator_cache):
        with open(snode_support_indicator_cache, 'r') as f:
            return f.read() == '1'
    try:
        with ti_FieldsBuilder() as builder:
            dummy = ti_field(ti.f64)
            builder.dense(ti.ij, (1, 1)).place(dummy)
            builder.finalize()
            _ = dummy.to_numpy()
            ret = True
    except RuntimeError:
        ret = False

    with open(snode_support_indicator_cache, 'w') as f:
        f.write('1' if ret else '0')

    return ret


def ti_ndarray_from(arr, sdim=0):
    """从其他数组类型创建Taichi ndarray

    Parameters
    ----------
    arr
        外部数组
    sdim
        结构维度，0代表Scalar，1代表Vector，2代表Matrix

    Returns
    -------
    ret
        返回结构维度为sdim的Taichi ndarray
    """
    ti_init_once()

    dt = to_taichi_type(arr.dtype)
    if sdim == 0:
        shape = arr.shape
    elif sdim == 1:
        dt = ti.types.vector(arr.shape[-1], dt)
        shape = arr.shape[:-1]
    elif sdim == 2:
        dt = ti.types.matrix(*arr.shape[-sdim:], dt)
        shape = arr.shape[:-sdim]
    else:
        raise ValueError(f"Unsupported structure dim: {sdim}")

    container = ti.ndarray(dtype=dt, shape=shape)
    typeinfo = type(arr)
    typename = typeinfo.__name__
    if typename == 'ndarray':
        container.from_numpy(arr)
    else:
        raise ValueError(f"Unsupported array type: {typename}")
    # taichi暂未为ndarray实现这些来源
    # elif typeinfo.__module__ == 'torch':
    #     container.from_torch(arr)
    # elif typeinfo.__module__ == 'paddle':
    #     container.from_paddle(arr)
    return container


@ti_func
def ti_index(i, j, stype: ti.template(), struct_size, array_size):
    """对struct array的索引，返回对应的1d index

    Parameters
    ----------
    i
        第i个struct
    j
        struct的第j个元素
    stype
        struct的内存布局类型
    struct_size
        struct的大小
    array_size
        array的大小

    Returns
    -------
    ret
        1d索引
    """

    if ti.static(stype == ti.Layout.AOS):
        return i * struct_size + j  # [x, y, z, x, y, z, ..., x, y, z]
    else:
        return j * array_size + i  # [x, x, ..., x, y, y, ..., y, z, z, ..., z]


def wrappable_for_ti_func(annotator):
    @functools.wraps(annotator)
    def wrapper(actual_func):
        # 用于处理ti.func或ti.kernel包装的函数
        # 这两类包装主要将实际功能函数添加到函数对象的__dict__中供调用
        if getattr(actual_func, '_is_taichi_function', False):
            setattr(actual_func, 'func', annotator(getattr(actual_func, 'func')))
            return actual_func
        elif getattr(actual_func, '_is_wrapped_kernel', False):
            setattr(actual_func, '_primal', annotator(getattr(actual_func, '_primal')))
            setattr(actual_func, '_adjoint', annotator(getattr(actual_func, '_adjoint')))
            return actual_func

        return annotator(actual_func)

    return wrapper


def check_contiguous(arr):
    typename = type(arr).__name__
    if typename == 'ndarray':
        import numpy as np
        if not arr.flags.contiguous:
            warnings.warn('Uncontiguous array detected.'
                          ' Copying to contiguous memory,'
                          ' which will consume extra memory.',
                          stacklevel=3)
            return np.ascontiguousarray(arr)
    else:
        # TODO: Torch、Paddle之类也可以检查？
        pass

    return arr


@wrappable_for_ti_func
def ensure_contiguous(func):
    def wrapper(*args, **kwargs):
        args = tuple(check_contiguous(arg) for arg in args)
        kwargs = {k: check_contiguous(v) for k, v in kwargs.items()}
        func(*args, **kwargs)

    return wrapper
