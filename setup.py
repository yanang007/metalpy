import contextlib
import re
import subprocess

from setuptools import setup
from pathlib import Path

from setuptools.command.sdist import sdist
from versioningit import get_cmdclasses


class RebuildSdistCommand(sdist):
    def exclude_untracked_source_files(self):
        """移除git未追踪的.py源文件
        TODO: 如果将来加入预生成的.py文件可能需要修改该函数
        """
        tracked_files = subprocess.check_output(['git', 'ls-files'], **{'text': 'utf-8'})
        tracked_files = tracked_files.strip().replace('\\', '/')  # 统一为反斜杠
        tracked_files = set(tracked_files.splitlines())

        files = self.filelist.files
        for i in range(len(files) - 1, 0, -1):
            if files[i].endswith('.py') and files[i].replace('\\', '/') not in tracked_files:
                print(f"removing untracked source file '{files[i]}'")
                del files[i]

    def rebuild_sdist(self):
        """因未知原因，打包出的sdist中路径会记录成绝对路径，因此在打包后添加一个workaround将路径转换为相对路径
        """
        gz_file = None
        for file in self.archive_files:
            if file.endswith('.tar.gz'):
                gz_file = Path(file)
                break

        if gz_file is not None:
            import gzip
            print(f'rebuilding {gz_file}')
            tar_file = gz_file.with_suffix('')
            g_file = gzip.GzipFile(gz_file)
            tar_file.write_bytes(g_file.read())
            g_file.close()

            with gzip.open(gz_file, "wb") as f_out:
                f_out.write(tar_file.read_bytes())

            tar_file.unlink()

    def make_distribution(self):
        self.exclude_untracked_source_files()

        super(RebuildSdistCommand, self).make_distribution()

        self.rebuild_sdist()


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
    root = 'https://github.com/yanang007/metalpy'
    if not pypi_readme_file.exists():
        long_description = readme_file.read_text(encoding='utf-8')
        long_description = re.compile(r'\(\.?[/|\\]?(.*?\.(md|rst))\)') \
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
        cmdclass=get_cmdclasses({'sdist': RebuildSdistCommand}),
        # Other arguments go here
    )
