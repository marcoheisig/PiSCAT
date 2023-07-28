from __future__ import annotations

import os
import pathlib
from typing import Union

from typing_extensions import Self

from piscat.io import FileWriter
from piscat.io.ffmpeg import FFmpegWriter
from piscat.io.numpy import NumpyWriter
from piscat.io.raw import RawWriter
from piscat.video.baseclass import precision_dtype, precision_next_power_of_two
from piscat.video.change_precision import Video_change_precision
from piscat.video.evaluation import Action, Batch, Chunk, compute_chunks

Path = Union[str, pathlib.Path]


class Video_to_file(Video_change_precision):
    def to_file(self, path: Path, /, overwrite: bool = False, flush: bool = False) -> Self:
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
        video = self
        writer: FileWriter
        if extension == "":
            raise ValueError(f"Couldn't determine the type of {path} (missing suffix).")
        elif extension == "raw":
            writer = RawWriter(path, video.shape, video.dtype)
        elif extension == "npy":
            video = video.change_precision(precision_next_power_of_two(video.precision))
            writer = NumpyWriter(path, video.shape, video.dtype)
        elif extension == "fits":
            raise NotImplementedError()
        elif extension == "fli":
            raise NotImplementedError()
        elif extension == "h5":
            raise NotImplementedError()
        else:
            video = video.change_precision(precision_next_power_of_two(video.precision))
            writer = FFmpegWriter(path, video.shape, video.dtype)
        target = Batch(Chunk((1,), precision_dtype(1)), 0, 1)
        position = 0
        for source in self.batches():
            count = source.stop - source.start
            WriterAction(target, source, writer, position)
            position += count
        compute_chunks([target.chunk])
        if not flush:
            return self
        return self  # TODO flush


class WriterAction(Action):
    writer: FileWriter
    position: int

    def __init__(self, target: Batch, source: Batch, writer: FileWriter, position: int):
        super().__init__([target], [source])
        self.writer = writer
        self.position = position

    def run(self):
        [(schunk, sstart, sstop)] = self.sources
        self.writer.write_chunk(schunk[sstart:sstop], self.position)
