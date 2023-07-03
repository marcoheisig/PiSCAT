from __future__ import annotations

import os
import pathlib
from typing import Union

from piscat.io import FileWriter
from piscat.io.ffmpeg import FFmpegWriter
from piscat.io.numpy import NumpyWriter
from piscat.io.raw import RawWriter
from piscat.video.baseclass import Video

Path = Union[str, pathlib.Path]


class Video_to_file(Video):
    def to_file(self, path: Path, /, overwrite: bool = False) -> Video:
        if isinstance(path, str):
            path = pathlib.Path(path)
        if path.exists():
            if overwrite:
                os.remove(path)
            else:
                raise ValueError(f"The file named {path} already exists.")
        suffix = path.suffix
        extension = suffix[1:] if suffix.startswith(".") else "npy"
        writer: FileWriter
        if extension == "":
            raise ValueError(f"Couldn't determine the type of {path} (missing suffix).")
        if extension == "raw":
            writer = RawWriter(path, self.shape, self.dtype)
        elif extension == "npy":
            writer = NumpyWriter(path, self.shape, self.dtype)
        elif extension == "fits":
            raise NotImplementedError()
        elif extension == "fli":
            raise NotImplementedError()
        elif extension == "h5":
            raise NotImplementedError()
        else:
            writer = FFmpegWriter(path, self.shape, self.dtype)
        position = 0
        for chunk, cstart, cstop in self.batches():
            count = cstop - cstart
            writer.write_chunk(chunk.data[cstart:cstop], position)
            position += count
        return self
