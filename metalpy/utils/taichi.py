import taichi as ti

from metalpy.mepa import LazyEvaluator

ti_args = {
    'arch': ti.cpu
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
