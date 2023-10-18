from __future__ import annotations

from abc import ABC

import dask.array as da
import numpy as np

# Video-related constants.

BYTES_PER_CHUNK = 2**26
MIN_PRECISION = 0
MAX_PRECISION = 64
FLOAT32 = np.dtype(np.float32)
FLOAT64 = np.dtype(np.float64)
FLOAT128 = np.dtype(np.float128)
INT8 = np.dtype(np.int8)
INT16 = np.dtype(np.int16)
INT32 = np.dtype(np.int32)
INT64 = np.dtype(np.int64)
UINT8 = np.dtype(np.uint8)
UINT16 = np.dtype(np.uint16)
UINT32 = np.dtype(np.uint32)
UINT64 = np.dtype(np.uint64)


# Video-related functions.


def dtype_bits(dtype: np.dtype):
    return dtype.itemsize * 8


def precision_next_power_of_two(precision: int) -> int:
    if precision < 0:
        raise ValueError(f"Invalid precision: {precision}")
    elif precision <= 8:
        return 8
    elif precision <= 16:
        return 16
    elif precision <= 32:
        return 32
    elif precision <= 64:
        return 64
    else:
        raise ValueError(f"Invalid precision: {precision}")


def precision_dtype(precision: int) -> np.dtype:
    if precision < 0:
        raise ValueError(f"Invalid precision: {precision}")
    elif precision <= 8:
        return UINT8
    elif precision <= 16:
        return UINT16
    elif precision <= 32:
        return UINT32
    elif precision <= 64:
        return UINT64
    else:
        raise ValueError(f"Invalid precision: {precision}")


def dtype_precision(dtype: np.dtype) -> int:
    if dtype == UINT8:
        return 8
    if dtype == UINT16:
        return 16
    if dtype == UINT32:
        return 32
    if dtype == UINT64:
        return 64
    raise ValueError(f"Invalid precision dtype: {dtype}")


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


# Video-related classes.


class Video(ABC):
    """
    The abstract base class of all video related classes.

    This class is inherited from by each class that defines video methods.  In
    the end, one class in __init__.py combines the behavior of all these classes
    into one.

    Parameters
    ----------

    shape : A tuple whose first element is the number of frames of the video,
    whose second element is the height of each frame, and whose third element is
    the width of each frame.

    precision : An integer between 0 and 64 (inclusive) that is the number of
    bits of information in each of the video's pixels.

    array : The underlying Dask array.  It has the same shape as the video, and
    an element type that is the smallest unsigned integer type able to hold
    values in the video's precision.
    """

    _array: da.Array
    _precision: int

    def __init__(
        self,
        array: da.Array,
        precision: int = 16,
    ):
        shape = array.shape
        if len(shape) != 3:
            raise ValueError("A video's shape must have rank three.")
        dtype = precision_dtype(precision)
        if array.dtype != dtype:
            raise ValueError(
                f"Expected a video of type {dtype}, but got one of type {array.dtype}."
            )
        self._precision = precision
        self._array = array

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the video as a (frames, height, width) tuple.
        """
        return self._array.shape

    @property
    def precision(self) -> int:
        """
        Return the number of bits of information per pixel of the video.
        """
        return self._precision

    @property
    def dtype(self) -> np.dtype:
        """
        Return the dtype of the video's array of pixels.
        """
        return precision_dtype(self.precision)

    def __len__(self):
        """
        Return the number of frames in the video.
        """
        return self.shape[0]

    def __repr__(self):
        return f"<Video shape={self.shape!r} precision={self.precision!r} id={id(self):#x}>"
