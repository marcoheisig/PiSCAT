from __future__ import annotations

import os
import pathlib
from typing import Union

Filename = Union[str, pathlib.Path]


def path_to_existing_file(filename: Filename) -> pathlib.Path:
    path = pathlib.Path(filename)
    if not path.exists():
        raise ValueError(f"No such file: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    return path


def path_to_new_file(filename: Filename, overwrite=False) -> pathlib.Path:
    path = pathlib.Path(filename)
    if path.exists():
        if not overwrite:
            raise ValueError(f"The file name {filename} already exists.")
        if not path.is_file():
            raise ValueError(f"Cannot override {filename}, which is not a file.")
        os.remove(path)
    return path
