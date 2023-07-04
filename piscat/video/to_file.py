from __future__ import annotations

import os
import pathlib
from typing import Union

from piscat.io import FileReader, FileWriter
from piscat.io.ffmpeg import FFmpegReader, FFmpegWriter
from piscat.io.numpy import NumpyReader, NumpyWriter
from piscat.io.raw import RawReader, RawWriter
from piscat.video.baseclass import Video
from piscat.video.evaluation import VideoOp
from piscat.video.from_file import get_reader_kernel

Path = Union[str, pathlib.Path]


class Video_to_file(Video):
    def to_file(self, path: Path, /, overwrite: bool = False, flush: bool = False) -> Video:
        """
        Write the video to the specified file.

        :param path: The name of the file where the video should be written to.
            The suffix of that path determines the nature of how the video is
            being stored.
        :param overwrite: Whether to overwrite any existing file at the
            specified path.
        :param flush: Whether to evict the video from main memory once it is
            written to the file.  Setting this file doesn't affect semantics,
            but flushing a video to a file frees main memory, at the price that
            the next reference to that video is much slower because it has to
            read that file back in.
        """
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
        elif extension == "raw":
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
        if not flush:
            return self
        # (optional) flush the video from main memory.
        reader: FileReader
        if extension == "raw":
            reader = RawReader(path, self.shape, self.dtype)
        elif extension == "npy":
            reader = NumpyReader(path)
        elif extension == "fits":
            raise NotImplementedError()
        elif extension == "fli":
            raise NotImplementedError()
        elif extension == "h5":
            raise NotImplementedError()
        else:
            reader = FFmpegReader(path)
        position = 0
        for batch in self.batches():
            (chunk, cstart, cstop) = batch
            count = cstop - cstart
            chunk._data = None
            kernel = get_reader_kernel(reader, position, position + count)
            VideoOp(kernel, [batch], [])
            position += count
        return self
