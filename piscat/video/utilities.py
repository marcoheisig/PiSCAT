from __future__ import annotations

import pathlib
from typing import Union

Filename = Union[str, pathlib.Path]


def canonicalize_file_path(filename: Filename) -> pathlib.Path:
    path = pathlib.Path(filename)
    if not path.exists():
        raise ValueError(f"No such file: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    return path
