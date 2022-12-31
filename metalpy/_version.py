__version__, __version_tuple__ = '__UNSPECIFIED__', '__UNSPECIFIED__'


def get_version():
    from pathlib import Path
    import versioningit
    if __version__ == '__UNSPECIFIED__':
        return versioningit.get_version(Path(__file__).parent.parent)
    else:
        return __version__


def get_version_tuple():
    from pathlib import Path
    import versioningit
    if __version_tuple__ == '__UNSPECIFIED__':
        version_tuple_str = versioningit.Versioningit\
            .from_project_dir(Path(__file__).parent.parent)\
            .do_template_fields(get_version(), None, None, None)['version_tuple']
        return eval(version_tuple_str)
    else:
        return __version_tuple__
