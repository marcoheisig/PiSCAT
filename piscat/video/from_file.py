from __future__ import annotations

import pathlib
from typing import Union

import dask
import dask.array as da
import filetype
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from piscat.io import FileReader
from piscat.io.ffmpeg import FFmpegReader
from piscat.io.numpy import NumpyReader
from piscat.video.baseclass import Video

Path = Union[str, pathlib.Path]


class Video_from_file(Video):
    @classmethod
    def from_raw_file(
        cls,
        path: Path,
        shape: tuple[int, int, int],
        dtype: npt.DTypeLike,
    ) -> Self:
        if isinstance(path, str):
            path = pathlib.Path(path)
        dtype = np.dtype(dtype)
        (f, h, w) = shape
        bytes_per_frame = h * w * dtype.itemsize
        bytes_per_chunk = 1024 * 1024 * 128
        frames_per_chunk = max(1, bytes_per_chunk // bytes_per_frame)
        load = dask.delayed(_chunk_from_raw_file)
        chunks = []
        for position in range(0, shape[0], frames_per_chunk):
            chunk_size = min(frames_per_chunk, shape[0] - position)
            offset = position * h * w * dtype.itemsize
            shape = (chunk_size, h, w)
            chunk = dask.array.from_delayed(
                load(path, offset=offset, shape=shape, dtype=dtype), shape=shape, dtype=dtype
            )
            chunks.append(chunk)
        return cls.from_array(da.concatenate(chunks, axis=0))

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
        raise NotImplementedError()


def _chunk_from_raw_file(filename, offset, shape, dtype):
    data = np.memmap(filename, mode="r", shape=shape, dtype=dtype, offset=offset)
