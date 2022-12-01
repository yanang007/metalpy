import os

import taichi as ti
from taichi.lang.util import to_taichi_type

from ..mepa import LazyEvaluator
from .file import make_cache_directory, make_cache_file

ti_cache_prefix = 'taichi_cache'

ti_args = {
    'arch': ti.cpu,
    'offline_cache': True,
    'offline_cache_file_path': make_cache_directory(ti_cache_prefix),
}
ti_inited = False


def ti_prepare(**kwargs):
    ti_args.update(kwargs)


def ti_init_once():
    global ti_inited
    if not ti_inited:
        ti.init(**ti_args)
        ti_inited = True


def ti_func(fn, is_real_function=None):
    if is_real_function is None:
        evaluator = LazyEvaluator(ti.func, fn)
    else:
        evaluator = LazyEvaluator(ti.func, fn, is_real_function)

    def lazy_evaluator_wrapper(*args, **kwargs):
        ti_init_once()
        return evaluator.get()(*args, **kwargs)

    return lazy_evaluator_wrapper


def ti_kernel(fn):
    evaluator = LazyEvaluator(ti.kernel, fn)

    def lazy_evaluator_wrapper(*args, **kwargs):
        ti_init_once()
        return evaluator.get()(*args, **kwargs)

    return lazy_evaluator_wrapper


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

    def __getattr__(self, name):
        return getattr(self.fields_builder, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


def ti_FieldsBuilder():
    ti_init_once()
    return WrappedFieldsBuilder()


def ti_test_snode_support() -> bool:
    """测试cuda版本是否支持SNode和FieldsBuilder

    如果不支持，则需要使用动态数组ti.ndarray来代替

    Returns
    -------
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
