from __future__ import annotations

import math

import dask.array as da
from typing_extensions import Self

from piscat.video.baseclass import MAX_PRECISION, dtype_precision, precision_dtype
from piscat.video.change_precision import Video_change_precision


class Video_rolling_average(Video_change_precision):
    def rolling_average(self, window_size: int) -> Self:
        rsum, precision, bits = _rolling_sum(self, window_size)
        # Multiply each sum by 2**bits before dividing, so that no precision is
        # lost in the process.  Be careful to avoid integer overflow.
        if precision + bits <= dtype_precision(rsum.dtype):
            array = (rsum << bits) // window_size
        else:
            (d, m) = da.divmod(rsum, window_size)
            array = (d << bits) + (m << bits) // window_size
        return type(self)(array, precision)

    def differential_rolling_average(self, window_size: int) -> Self:
        ravg = self.rolling_average(window_size)
        if ravg.precision == MAX_PRECISION:
            p = MAX_PRECISION
            s = ravg._array >> 1
        else:
            p = 1 + ravg.precision
            s = ravg._array
        a = s[:-window_size]
        b = s[window_size:]
        base = 2 ** (p - 1) - 1
        dtype = precision_dtype(p)
        x = da.add(base, a, dtype=dtype)
        y = da.subtract(x, b, dtype=dtype)
        return type(ravg)(y, p)


def _rolling_sum(video: Video_change_precision, window_size: int) -> tuple[da.Array, int, int]:
    if window_size < 1:
        raise ValueError(f"Invalid window size: {window_size}")
    if window_size == 1:
        return (video._array, video.precision, 0)
    if window_size > len(video):
        raise ValueError(f"Invalid window size {window_size} for video of length {len(video)}.")
    # Ensure that none of the calculations will overflow.
    bits = math.ceil(math.log2(window_size))
    video = video.change_precision(min(video.precision, MAX_PRECISION - bits))
    array = video._array
    precision = video.precision + bits
    # Dask overlapping doesn't support overlapping that is larger than two
    # consecutive chunks minus one.  Rechunk if necessary.
    sizes = array.chunks[0]
    if len(sizes) > 1:
        delta = min(a + b for a, b in zip(sizes[:-1], sizes[1:]))
        if 2 * window_size > delta:
            array = array.rechunk((2 * window_size, "auto", "auto"))
    g1 = da.overlap.overlap(array, depth={0: (0, window_size - 1)}, boundary={0: "none"})
    c1 = g1.chunks
    c2 = (tuple(s - window_size + 1 for s in c1[0]), c1[1], c1[2])
    g2 = g1.map_blocks(
        _blockwise_rolling_sum, dtype=precision_dtype(precision), w=window_size, chunks=c2
    )
    return (g2, precision, bits)


def _blockwise_rolling_sum(x, w, block_info):
    """
    Compute the rolling sum within a block X with window size W, where W is
    assumed to be a fraction of the total size of X.
    """
    dtype = block_info[None]["dtype"]
    size = len(x) - w + 1
    result = type(x)((size, *x.shape[1:]), dtype=dtype)
    if w <= 3:
        result[:] = 0
        for p in range(0, w):
            result += x[p : size + p]
    else:
        result[0] = x[:w].sum(axis=0)
        for p in range(1, size):
            result[p] = result[p - 1] - x[p - 1] + x[p + w - 1]
    return result
