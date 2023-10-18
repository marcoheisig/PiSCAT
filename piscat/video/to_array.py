from __future__ import annotations

import dask.array as da

from piscat.video.baseclass import FLOAT32, FLOAT64, dtype_bits
from piscat.video.change_precision import Video_change_precision


class Video_to_array(Video_change_precision):
    def to_array(self, dtype=None):
        return array_from_video_data(self._array, self.precision, dtype).compute()

    def __array__(self):
        return self.to_array()


def array_from_video_data(array: da.Array, precision: int, dtype) -> da.Array:
    dtype = array.dtype if dtype is None else dtype
    if dtype == FLOAT32:
        return array.astype(dtype) / (2**precision - 1)
    elif dtype == FLOAT64:
        return array.astype(dtype) / (2**precision - 1)
    elif dtype.kind == "u":
        shift = dtype_bits(dtype) - precision
        if shift == 0:
            return array
        elif shift > 0:
            return array << +shift
        else:
            return array >> -shift
    elif dtype.kind == "i":
        bits = dtype_bits(dtype)
        shift = bits - precision
        if shift == 0:
            shifted = array
        elif shift > 0:
            shifted = array << +shift
        else:
            shifted = array >> -shift
        return (shifted - 2 ** (bits - 1)).view(dtype)
    else:
        raise ValueError(f"Invalid dtype: {dtype}")
