"""
用于启动一个单一主线程的dask worker，防止一些存在线程安全问题的库出现问题
"""

import click
from distributed import Worker
from distributed.cli.dask_worker import main
from concurrent.futures import Future, Executor


class DummyWorker(Worker):
    def __init__(self, *args, **kwargs):
        print('Dummy啦')
        super().__init__(*args, executor=DummyLinearExecutor(), **kwargs)


class DummyLinearExecutor(Executor):
    def __init__(self):
        pass

    def submit(self, fn, /, *args, **kwargs):
        fut = Future()
        result = fn(*args, **kwargs)
        fut.set_result(result)

        return fut


if __name__ == "__main__":
    use_dummy_executor = True  # 是否使用上文实现的单线程执行器，否则使用dask的offload执行器
    if use_dummy_executor:
        click.option(
            "--worker-class",
            type=str,
            default='startup.DummyWorker',
            show_default=True,
            help="Default executor.",
        )(main)()
    else:
        click.option(
            "--executor",
            type=str,
            default="offload",
            show_default=True,
            help="Default executor.",
        )(main)()
