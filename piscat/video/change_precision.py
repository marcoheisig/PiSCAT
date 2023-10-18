from __future__ import annotations

import operator

from dask.array.core import elemwise

from piscat.video.baseclass import MAX_PRECISION, MIN_PRECISION, Video, precision_dtype


class Video_change_precision(Video):
    def change_precision(self, precision: int):
        dtype = precision_dtype(precision)
        if precision == self.precision:
            return self
        elif MIN_PRECISION <= precision < self.precision:
            shift = self.precision - precision
            array = elemwise(operator.rshift, self._array, shift, dtype=dtype)
            return type(self)(array, precision)
        elif MAX_PRECISION >= precision > self.precision:
            shift = precision - self.precision
            array = elemwise(operator.lshift, self._array.astype(dtype), shift)
            return type(self)(array, precision)
        else:
            raise TypeError(f"Invalid video precision: {precision}")
