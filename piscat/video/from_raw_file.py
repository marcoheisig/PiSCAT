from __future__ import annotations

import pathlib
from typing import Union

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from piscat.video.from_array import Video_from_array
from piscat.video.utilities import Filename, canonicalize_file_path

Path = Union[str, pathlib.Path]


class Video_from_raw_file(Video_from_array):
    @classmethod
    def from_raw_file(
        cls,
        filename: Filename,
        shape: tuple[int, int, int],
        dtype: npt.DTypeLike,
    ) -> Self:
        path = canonicalize_file_path(filename)
        dtype = np.dtype(dtype)
        (f, h, w) = shape
        bytes_per_frame = h * w * dtype.itemsize
        bytes_per_chunk = 1024 * 1024 * 128
        frames_per_chunk = max(1, bytes_per_chunk // bytes_per_frame)
        load = dask.delayed(_chunk_from_raw_file)
        chunks = []
        for position in range(0, f, frames_per_chunk):
            chunk_size = min(frames_per_chunk, shape[0] - position)
            offset = position * h * w * dtype.itemsize
            shape = (chunk_size, h, w)
            chunk = dask.array.from_delayed(
                load(path, offset=offset, shape=shape, dtype=dtype), shape=shape, dtype=dtype
            )
            chunks.append(chunk)
        return cls.from_array(da.concatenate(chunks, axis=0))


def _chunk_from_raw_file(filename, offset, shape, dtype):
    return np.memmap(filename, mode="r", shape=shape, dtype=dtype, offset=offset)
