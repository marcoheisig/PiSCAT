from __future__ import annotations

import pathlib
from typing import Union

import filetype
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from piscat.io import FileReader
from piscat.io.ffmpeg import FFmpegReader
from piscat.io.numpy import NumpyReader
from piscat.io.raw import RawReader
from piscat.video.actions import dtype_decoder_and_precision
from piscat.video.baseclass import Video, precision_dtype
from piscat.video.evaluation import Action, Batch, Chunk

Path = Union[str, pathlib.Path]


class Video_from_file(Video):
    @classmethod
    def from_file(cls, path: Path) -> Self:
        """
        Return a video whose contents are read in from a file.

        :param path: The path to the video to read.
        :param format: The format of the video.  Valid options are "raw", "tif",
            "fits", "fli", "h5", "npy", and any format supported by ffmpeg such
            as "avi", "webm", and "mp4".  If no format is specified, an attempt
            is made to infer it from the suffix of the supplied path, or by
            looking at the file itself.
        :param shape: The expected shape of the video being read in.  If
            supplied, an error is raised if the video's shape differs from its
            expectation.
        :param dtype: The element type of the resulting video.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"No such file: {path}")
        # Determine the file extension.
        if path.suffix and path.suffix.startswith("."):
            extension = path.suffix[1:]
        else:
            extension = filetype.guess_extension(path)
        # Create an appropriate video reader.
        reader: FileReader
        if extension == "raw":
            raise RuntimeError("Please use Video.from_raw_file to load raw videos.")
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
        # Create the video.
        shape = reader.shape
        file_dtype = reader.dtype
        decoder, precision = dtype_decoder_and_precision(file_dtype)
        video_dtype = precision_dtype(precision)
        chunk_size = Video.plan_chunk_size(shape, precision)
        (f, h, w) = shape
        chunks = []
        for start in range(0, f, chunk_size):
            stop = min(f, start + chunk_size)
            count = stop - start
            tmp = Batch(Chunk((chunk_size, h, w), file_dtype), 0, count)
            ReaderAction(tmp, reader, start, stop)
            target = Batch(Chunk((chunk_size, h, w), video_dtype), 0, count)
            decoder(target, tmp)
            chunks.append(target.chunk)
        return cls(chunks, shape, precision=precision)

    @classmethod
    def from_raw_file(
        cls,
        path: Path,
        shape: tuple[int, int, int],
        dtype: npt.DTypeLike,
    ) -> Self:
        if isinstance(path, str):
            path = pathlib.Path(path)
        file_dtype = np.dtype(dtype)
        decoder, precision = dtype_decoder_and_precision(file_dtype)
        video_dtype = precision_dtype(precision)
        reader = RawReader(path, shape, file_dtype)
        chunk_size = Video.plan_chunk_size(shape, precision)
        (f, h, w) = shape
        chunks = []
        for start in range(0, f, chunk_size):
            stop = min(f, start + chunk_size)
            count = stop - start
            tmp = Batch(Chunk((chunk_size, h, w), file_dtype), 0, count)
            ReaderAction(tmp, reader, start, stop)
            target = Batch(Chunk((chunk_size, h, w), video_dtype), 0, count)
            decoder(target, tmp)
            chunks.append(target.chunk)
        return cls(chunks, shape, precision=precision)


class ReaderAction(Action):
    reader: FileReader
    start: int
    stop: int

    def __init__(self, target: Batch, reader: FileReader, start: int, stop: int):
        super().__init__([target], [])
        self.reader = reader
        self.start = start
        self.stop = stop

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        self.reader.read_chunk(tchunk.data[tstart:tstop], self.start, self.stop)
