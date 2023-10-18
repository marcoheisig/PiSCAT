from __future__ import annotations

import dask.array as da

from piscat.video.baseclass import FLOAT32, FLOAT64, dtype_bits
from piscat.video.change_precision import Video_change_precision


class Video_to_array(Video_change_precision):
    def to_array(self, dtype=None):
        return array_from_video_data(self._array, self.precision, dtype)

    def __array__(self):
        return self.to_array()


def array_from_video_data(array, precision, dtype) -> da.Array:
    dtype = array.dtype if dtype is None else dtype
    if dtype == FLOAT32:
        return (array.astype(dtype) / (2**precision - 1)).compute()
    elif dtype == FLOAT64:
        return (array.astype(dtype) / (2**precision - 1)).compute()
    elif dtype.kind == "u":
        shift = dtype_bits(dtype) - precision
        if shift == 0:
            return array.compute()
        elif shift > 0:
            return (array << +shift).compute()
        else:
            return (array >> -shift).compute()
    elif dtype.kind == "i":
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid dtype: {dtype}")
