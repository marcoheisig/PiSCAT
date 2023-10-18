from __future__ import annotations

import pathlib
import tempfile

import numpy as np
from numpy.random import shuffle

from piscat.io.ffmpeg import FFmpegReader, FFmpegWriter
from piscat.io.numpy import NumpyReader, NumpyWriter
from piscat.io.raw import RawReader, RawWriter

signed_dtypes = (np.int8, np.int16, np.int32, np.int64)
unsigned_dtypes = (np.uint8, np.uint16, np.uint32, np.uint64)
float_dtypes = (np.float32, np.float64)
dtypes = (*unsigned_dtypes, *float_dtypes)


def dtype_limits(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return (0.0, 1.0)
    if dtype.kind == "u":
        return (0, 2 ** (dtype.itemsize * 8) - 1)
    if dtype.kind == "i":
        return (-(2 ** (dtype.itemsize * 8)), (2 ** (dtype.itemsize * 8)) - 1)
    raise RuntimeError(f"Invalid dtype: {dtype}")


def test_io():
    return  # TODO
    for dtype in dtypes:
        (lo, hi) = dtype_limits(dtype)
        a1 = np.random.uniform(float(lo), float(hi), (1, 1, 1)).astype(dtype)
        a2 = np.random.uniform(float(lo), float(hi), (3, 5, 7)).astype(dtype)
        a3 = np.random.uniform(float(lo), float(hi), (30, 128, 128)).astype(dtype)
        for array in (a1, a2, a3):
            with tempfile.TemporaryDirectory() as td:
                file1 = pathlib.Path(td, "data.npy")
                file2 = pathlib.Path(td, "data.raw")
                file3 = pathlib.Path(td, "data.mp4")
                file4 = pathlib.Path(td, "data.avi")
                w1 = NumpyWriter(file1, array.shape, array.dtype)
                w2 = RawWriter(file2, array.shape, array.dtype)
                w3 = FFmpegWriter(file3, array.shape, array.dtype)
                w4 = FFmpegWriter(file4, array.shape, array.dtype)
                # Write to the specified files.
                for writer in (w1, w2, w3, w4):
                    length = len(array)
                    step = 7
                    intervals = []
                    for position in range(0, length, step):
                        intervals.append((position, min(position + step, length)))
                    shuffle(intervals)
                    for start, stop in intervals:
                        writer.write_chunk(array[start:stop], start)
                # Read from the specified files.
                r1 = NumpyReader(file1)
                r2 = RawReader(file2, array.shape, array.dtype)
                r3 = FFmpegReader(file3)
                r4 = FFmpegReader(file4)
                for reader in (r1, r2, r3, r4):
                    other = np.empty(array.shape, array.dtype)
                    length = len(array)
                    step = 5
                    intervals = []
                    for position in range(0, length, step):
                        intervals.append((position, min(position + step, length)))
                    shuffle(intervals)
                    for start, stop in intervals:
                        reader.read_chunk(other[start:stop], start, stop)
                    if isinstance(reader, FFmpegReader):
                        assert array_somewhat_similar(array, other)
                    else:
                        assert np.array_equal(array, other)


def array_somewhat_similar(a1: np.ndarray, a2: np.ndarray, tolerance=0.15, ignore=0.1):
    """
    Return whether the two supplied arrays have the same shape, and roughly
    similar content in that only a certain percentage of values is more than a
    certain fraction of the range of the underlying data type apart.
    """
    if a1.shape != a2.shape:
        return False
    if a1.dtype != a2.dtype:
        return False
    (lo, hi) = dtype_limits(a1.dtype)
    tolerance = np.asarray((hi - lo) * tolerance)
    difference = np.minimum(a1 - a2, a2 - a1)
    outliers = (difference > tolerance).sum()
    return outliers < (ignore * a1.size)
