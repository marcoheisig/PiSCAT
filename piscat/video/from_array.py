from __future__ import annotations

from math import ceil, log2

import dask.array as da
import numpy as np
from typing_extensions import Self

from piscat.video.baseclass import FLOAT32, FLOAT64, FLOAT128, Video, dtype_bits, precision_dtype


class Video_from_array(Video):
    @classmethod
    def from_array(cls, array, /, lo=None, hi=None, precision: int | None = None) -> Self:
        """
        Return a video whose contents are derived from the supplied array.

        Parameters
        ----------

        lo : A number that marks the lower end of the relevant spectrum.  All
        array elements less than this value are counted as having an intensity
        of zero in the resulting video.  If not supplied, a suitable default is
        derived form the supplied array's dtype: For floating-point numbers the
        default value is zero, and for signed or unsigned integer types the
        default is the lowest integer of that type.

        hi : A number that marks the upper end of the relevant spectrum.  All
        array elements greater than this value are counted as having maximum
        intensity in the resulting video.  If not supplied, a suitable default
        is derived form the supplied array's dtype: For floating-point numbers
        the default value is one, and for signed or unsigned integer types the
        default is the highest integer of that type.

        precision: An integer that is the desired precision of the resulting
        video.  Defaults to 24 for single-precision floating-point numbers, 53
        for double-precision floating-point numbers, and the ceiling of the
        two's logarithm of the distance between hi and lo for integers.
        """
        if not isinstance(array, da.Array):
            array = da.from_array(array)
        shape = array.shape
        if len(shape) != 3:
            raise ValueError("Can only turn three-dimensional arrays into videos.")
        dtype = array.dtype
        kind = dtype.kind
        bits = dtype_bits(dtype)
        # Canonicalize lo, hi, and precision.
        if kind == "u":
            lo = 0 if lo is None else lo
            hi = (2**bits) - 1 if hi is None else hi
            precision = ceil(log2(1 + hi - lo)) if precision is None else precision
        elif kind == "i":
            lo = -(2 ** (bits - 1)) if lo is None else lo
            hi = (2 ** (bits - 1)) - 1 if hi is None else hi
            precision = ceil(log2(1 + hi - lo)) if precision is None else precision
        elif dtype == FLOAT32:
            lo = 0.0 if lo is None else lo
            hi = 1.0 if hi is None else hi
            precision = 24 if precision is None else precision
        elif dtype == FLOAT64:
            lo = 0.0 if lo is None else lo
            hi = 1.0 if hi is None else hi
            precision = 53 if precision is None else precision
        elif dtype == FLOAT128:
            lo = 0.0 if lo is None else lo
            hi = 1.0 if hi is None else hi
            precision = 53 if precision is None else precision
        else:
            raise ValueError(f"Cannot convert {dtype} arrays to videos.")
        if hi <= lo:
            raise ValueError(f"Invalid bounds: [{lo}, {hi}]")
        data = array.clip(lo, hi) - lo
        numerator = 2**precision - 1
        denominator = hi - lo
        print((lo, hi, precision, numerator, denominator))
        if numerator == denominator:
            result = data
        elif precision > 53:
            factor = np.longdouble(numerator) / np.longdouble(denominator)
            result = data * factor
        else:
            factor = numerator / denominator
            result = data * factor
        return cls(result.astype(precision_dtype(precision)), precision)


def video_data_from_array(array, lo=None, hi=None, precision=None) -> da.Array:
    pass
