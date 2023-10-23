from __future__ import annotations

import os.path

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from piscat.video.from_array import Video_from_array
from piscat.video.utilities import Filename, path_to_existing_file


class Video_from_raw_file(Video_from_array):
    @classmethod
    def from_raw_file(
        cls,
        filename: Filename,
        shape: tuple[int, int, int] | tuple[int, int],
        dtype: npt.DTypeLike,
    ) -> Self:
        path = path_to_existing_file(filename)
        nbytes = os.path.getsize(path)
        dtype = np.dtype(dtype)
        if len(shape) == 3:
            (f, h, w) = shape
        elif len(shape) == 2:
            (h, w) = shape
            f = nbytes // (h * w * dtype.itemsize)
        else:
            raise ValueError(f"Invalid raw video shape: {shape}")
        bytes_per_frame = h * w * dtype.itemsize
        bytes_per_chunk = 1024 * 1024 * 128
        frames_per_chunk = max(1, bytes_per_chunk // bytes_per_frame)
        load = dask.delayed(_chunk_from_raw_file)
        chunks = []
        for position in range(0, f, frames_per_chunk):
            chunk_size = min(frames_per_chunk, f - position)
            offset = position * h * w * dtype.itemsize
            shape = (chunk_size, h, w)
            chunk = dask.array.from_delayed(
                load(path, offset=offset, shape=shape, dtype=dtype), shape=shape, dtype=dtype
            )
            chunks.append(chunk)
        return cls.from_array(da.concatenate(chunks, axis=0))


def _chunk_from_raw_file(filename, offset, shape, dtype):
    return np.memmap(filename, mode="r", shape=shape, dtype=dtype, offset=offset)
