import gzip
import os

from setuptools import build_meta as _orig

prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_wheel = _orig.build_wheel
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist


def build_sdist(*args, **kwargs):
    """因未知原因，打包出的sdist中路径会记录成绝对路径，因此在打包后添加一个workaround将路径转换为相对路径"""

    ret = _orig.build_sdist(*args, **kwargs)

    dist_path = r'./dist'
    gz_name = None
    for file in os.listdir(dist_path):
        if file.endswith('.tar.gz'):
            gz_name = os.path.join(dist_path, file)
            break

    if gz_name is not None:
        tar_name = gz_name.replace(".gz", "")
        g_file = gzip.GzipFile(gz_name)
        open(tar_name, "wb+").write(g_file.read())
        g_file.close()

        with open(tar_name, "rb") as f_in:
            with gzip.open(gz_name, "wb") as f_out:
                f_out.write(f_in.read())

        os.remove(tar_name)

    return ret