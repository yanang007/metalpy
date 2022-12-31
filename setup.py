import contextlib
import re

from setuptools import setup
from pathlib import Path

from versioningit import get_cmdclasses


@contextlib.contextmanager
def generate_pypi_readme():
    """从仓库版README.md中生成适用于pypi的README文件

    Notes
    -----
        这是对当前setuptools系统不支持pyproject.toml联合使用setup.py动态添加long_description (readme) 的workaround
        该workaround的特殊处理包括：
            在MANIFEST.in中手动排除会被自动包含的README.md，并手动包含生成的文件README-pypi.md
            在pyproject.toml中指定readme = "README-pypi.md"

        注意事项：
            README.md中：
            所有.md或.rst结尾的引用会被替换为github地址的引用
            所有./都会被替换为github地址的raw对象地址引用
    """
    readme_file = Path("README.md")
    pypi_readme_file = Path('README-pypi.md')
    root = 'github.com/yanang007/metalpy'
    if not pypi_readme_file.exists():
        long_description = readme_file.read_text(encoding='utf-8')
        long_description = re.compile(r'\(\.?[/|\\]?(.*?\.(md|rst))\)')\
            .sub(rf'({root}/tree/main/\1)', long_description)
        long_description = long_description.replace('./', f'{root}/raw/main/')

        pypi_readme_file.write_text(long_description, encoding='utf-8')

    try:
        yield
    finally:
        if pypi_readme_file.exists() and readme_file.exists():
            # 如果两个README.md都存在，则应当清理掉临时生成的pypi-README.md
            # - 这种情况包含在项目路径下构建sdist的阶段
            # 否则如果只存在pypi-README.md，则不应该清理pypi-README.md
            # - 这种情况包含从sdist构建wheel的阶段，会创建一个只包含sdist中文件的临时目录，由于README.md被排除，所以此时不存在README.md
            pypi_readme_file.unlink()


with generate_pypi_readme():
    setup(
        cmdclass=get_cmdclasses(),
        # Other arguments go here
    )
