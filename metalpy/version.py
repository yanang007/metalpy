from typing import NamedTuple

from ._version import get_version, get_version_tuple

__version__ = get_version()
version_tuple = get_version_tuple()
full_version = __version__
base_version = NamedTuple('ShortVersion', major=int, minor=int, patch=int)(*version_tuple[:3])
short_version = '.'.join((str(i) for i in version_tuple[:3]))
is_release = len(version_tuple) == 3

post_distance = 0
revision_prefix = 'post'
if not is_release:
    for ver_desc in version_tuple[3:]:
        if ver_desc.startswith(revision_prefix):
            post_distance = int(version_tuple[3][len(revision_prefix):])
            break

del get_version, get_version_tuple, revision_prefix
