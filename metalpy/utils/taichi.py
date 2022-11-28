import taichi as ti

from metalpy.mepa import LazyEvaluator

ti_args = {
    'arch': ti.cpu
}
ti_inited = False


def ti_prepare(**kwargs):
    ti_args.update(kwargs)


def ti_init_once():
    if not ti_inited:
        ti.init(**ti_args)


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
