import taichi as ti

ti.init(arch=ti.cpu)


def ti_func(fn, is_real_function=None):
    if is_real_function is None:
        return ti.func(fn)
    else:
        return ti.func(fn, is_real_function)


def ti_kernel(fn):
    return ti.kernel(fn)
