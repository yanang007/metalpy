from typing import NamedTuple

from ._version import get_version, get_version_tuple

__version__ = get_version()
version_tuple = get_version_tuple()
full_version = __version__
short_version = NamedTuple('ShortVersion', major=str, minor=str, patch=str)(*version_tuple[:3])
is_release = len(version_tuple) == 3

del get_version, get_version_tuple
