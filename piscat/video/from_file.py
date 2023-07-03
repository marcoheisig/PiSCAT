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
from piscat.video.baseclass import Video
from piscat.video.evaluation import Batch, Batches, Kernel, VideoChunk, VideoOp

Path = Union[str, pathlib.Path]


class Video_from_file(Video):
    @classmethod
    def from_file(cls, path: Path, /, chunk_size: int | None = None) -> Self:
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
        dtype = reader.dtype
        if chunk_size is None:
            chunk_size = Video.plan_chunk_size(shape, dtype)
        (f, h, w) = shape
        chunks = []
        for start in range(0, f, chunk_size):
            stop = min(f, start + chunk_size)
            count = stop - start
            chunk = VideoChunk((chunk_size, h, w), dtype)
            kernel = get_reader_kernel(reader, start, stop)
            VideoOp(kernel, [Batch(chunk, 0, count)], [])
            chunks.append(chunk)
        return cls(chunks, length=shape[0])

    @classmethod
    def from_raw_file(
        cls,
        path: Path,
        shape: tuple[int, int, int],
        dtype: npt.DTypeLike,
        /,
        chunk_size: int | None = None,
    ) -> Self:
        if isinstance(path, str):
            path = pathlib.Path(path)
        dtype = np.dtype(dtype)
        reader = RawReader(path, shape, np.dtype(dtype))
        if chunk_size is None:
            chunk_size = Video.plan_chunk_size(shape, dtype)
        (f, h, w) = shape
        chunks = []
        for start in range(0, f, chunk_size):
            stop = min(f, start + chunk_size)
            count = stop - start
            chunk = VideoChunk((chunk_size, h, w), dtype)
            kernel = get_reader_kernel(reader, start, stop)
            VideoOp(kernel, [Batch(chunk, 0, count)], [])
            chunks.append(chunk)
        return cls(chunks, length=shape[0])


def get_reader_kernel(reader: FileReader, start: int, stop: int) -> Kernel:
    def kernel(targets: Batches, sources: Batches) -> None:
        assert len(targets) == 1
        assert len(sources) == 0
        (chunk, cstart, cstop) = targets[0]
        assert (stop - start) == (cstop - cstart)
        reader.read_chunk(chunk.data[cstart:cstop], start, stop)

    return kernel
