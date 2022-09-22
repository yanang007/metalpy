import importlib
import os
import shutil

import sys
import uuid
import zipfile

from dask.utils import tmpfile
from distributed import WorkerPlugin, NannyPlugin
from distributed.utils import logger


def configure_dask_client(client, extra_paths=None, excludes=None):
    class UploadModules(NannyPlugin):
        def __init__(self, identifier=None):
            self.modules = []

            if identifier is None:
                self.identifier = str(uuid.uuid4())
            else:
                self.identifier = str(identifier)

        def add(self, module_name, path):
            self.modules.append((module_name, path))

        def compact(self):
            with tmpfile() as fn:
                with zipfile.ZipFile(fn, "w") as z:
                    for module_name, path in self.modules:
                        module_path = module_name.replace(".", "/")
                        path = os.path.abspath(path).replace("\\", "/")
                        index = path.find(module_path)
                        archive_name = path[index:]

                        z.write(path, archive_name)

                with open(fn, "rb") as f:
                    self.data = f.read()

        def get_reloader(self):
            return ReloadModules(self.modules)

        async def setup(self, nanny):
            target = nanny

            fn = os.path.join(target.local_directory, f"tmp-{self.identifier}.zip")
            if not os.path.exists(fn):
                with open(fn, "wb") as f:
                    f.write(self.data)

                import zipfile

                base_dir = target.local_directory
                if 'dask-worker-space' not in target.local_directory:
                    # 在不同配置下nanny的工作目录可能在dask-worker-space的更上一级目录
                    # 具体来说是如果通过 --local-directory 指定了worker的工作路径，nanny的工作路径就会变为
                    # %working_dir% 而不是之前的 %working_dir%/dask-worker-space
                    # 此时应该切进 dask-worker-space 以保持一致
                    base_dir = os.path.join(base_dir, 'dask-worker-space')

                check_directory_flag = False
                for content in os.listdir(base_dir):
                    if 'worker-' in content:
                        check_directory_flag = True
                        break
                if not check_directory_flag:
                    logger.warning('Wrong nanny working directory: %s', base_dir)

                path_to_scripts = os.path.join(base_dir, "scripts")

                if os.path.exists(path_to_scripts):
                    shutil.rmtree(path_to_scripts)

                with zipfile.ZipFile(fn) as z:
                    z.extractall(path=path_to_scripts)

                os.remove(fn)

    class ReloadModules(WorkerPlugin):
        def __init__(self, modules):
            self.modules = modules

        async def setup(self, worker=None):
            target = worker
            base_dir = os.path.abspath(os.path.join(target.local_directory, '../..'))
            if 'dask-worker-space' not in base_dir:
                # 同上，如果nanny的工作路径不在dask-worker-space下，那么worker的工作路径似乎也不会在dask-worker-space下
                base_dir = os.path.join(base_dir, 'dask-worker-space')

            path_to_scripts = os.path.join(base_dir, "scripts")

            print(path_to_scripts)

            importlib.invalidate_caches()
            sys.path.insert(0, path_to_scripts)

            # import modules in the scripts folder
            for module_name, path in self.modules:
                logger.info("Reloading module %s from .py file", module_name)
                importlib.reload(importlib.import_module(module_name))

            sys.path.remove(path_to_scripts)

    if extra_paths is None:
        paths = []
    else:
        paths = [os.path.abspath(p) for p in extra_paths]

    if excludes is None:
        excludes = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_module_location = __name__.split('.')

    while os.path.split(current_dir)[1] in current_module_location:
        current_dir = os.path.split(current_dir)[0]

    paths.append(current_dir)
    paths.append(os.path.abspath('./'))

    # 将脚本文件上传到服务器
    modules = list(sys.modules.items())
    plugin = UploadModules(uuid.uuid4())

    for name, module in modules:
        try:
            file_path = module.__spec__.origin
        except AttributeError:
            continue

        if file_path is None:
            continue

        required = False
        for path in paths:
            if path not in os.path.dirname(file_path):
                continue
            else:
                required = True
                break

        if not required:
            continue

        for exclude in excludes:
            if hasattr(exclude, '__call__'):
                required = not exclude(name)
            else:
                required = exclude not in name

            if not required:
                break

        if not required:
            continue

        plugin.add(name, file_path)

    plugin.compact()

    client.register_worker_plugin(plugin, nanny=True, name='UploadUserModules')
    client.register_worker_plugin(plugin.get_reloader(), name='ReloadUserModules')
    client.unregister_worker_plugin(name='ReloadUserModules')
    client.unregister_worker_plugin(nanny=True, name='UploadUserModules')
