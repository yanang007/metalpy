import inspect
import warnings
from typing import Callable


def test_rewrote_func_compatibility():
    from taichi.lang.kernel_impl import _kernel_impl
    from taichi import func

    funcs: dict[str, [str, Callable]] = {
        'ti_kernel': (fstr_kernel_impl, _kernel_impl),
        'ti_func': (fstr_ti_func, func),
    }

    mismatched = []

    for func_name, (snapshot, target_func) in funcs.items():
        if inspect.getsource(target_func).strip() != snapshot.strip():
            mismatched.append(func_name)

    for func_name in mismatched:
        warnings.warn(f"`{func_name}`'s target function has different body from snapshot."
                      f" Consider update `{func_name}` or its snapshot?")

    assert len(mismatched) == 0


fstr_kernel_impl = """
def _kernel_impl(_func, level_of_class_stackframe, verbose=False):
    # Can decorators determine if a function is being defined inside a class?
    # https://stackoverflow.com/a/8793684/12003165
    is_classkernel = _inside_class(level_of_class_stackframe + 1)

    if verbose:
        print(f'kernel={_func.__name__} is_classkernel={is_classkernel}')
    primal = Kernel(_func,
                    autodiff_mode=AutodiffMode.NONE,
                    _classkernel=is_classkernel)
    adjoint = Kernel(_func,
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
        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            # If we reach here (we should never), it means the class is not decorated
            # with @ti.data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            assert not hasattr(clsobj, '_data_oriented')
            raise TaichiSyntaxError(
                f'Please decorate class {clsobj.__name__} with @ti.data_oriented'
            )
    else:

        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            try:
                return primal(*args, **kwargs)
            except (TaichiCompilationError, TaichiRuntimeError) as e:
                raise type(e)('\\n' + str(e)) from None

        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    return wrapped
"""
