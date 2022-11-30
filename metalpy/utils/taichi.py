import taichi as ti

from metalpy.mepa import LazyEvaluator
from metalpy.utils.file import make_cache_directory

ti_args = {
    'arch': ti.cpu,
    'offline_cache': True,
    'offline_cache_file_path': make_cache_directory('taichi_cache'),
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

    try:
        with ti_FieldsBuilder() as builder:
            dummy = ti_field(ti.f64)
            builder.dense(ti.ij, (1, 1)).place(dummy)
            builder.finalize()
            _ = dummy.to_numpy()
    except RuntimeError:
        return False

    return True
