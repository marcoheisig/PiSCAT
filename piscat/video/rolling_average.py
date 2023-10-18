from __future__ import annotations

import math

import dask.array as da
from typing_extensions import Self

from piscat.video.baseclass import MAX_PRECISION, precision_dtype
from piscat.video.change_precision import Video_change_precision

# Some pondering about efficient rolling averages:
#
# The goal is to turn a video V with N frames into one with N - W + 1 frames,
# where W is the supplied window size.  Each frame K of the resulting video is
# the sum of the frames V(K), ..., V(K+W-1) divided by W.  We break this
# procedure up into two steps: In the first step, we compute a video of the sums
# S(K) = K, ..., K+W-1.  In the second step, we divide all values in that video
# of sums by W.  The second step is embarrassingly parallel, so the main
# challenge is to implement the first step efficiently.  There are three good
# options for computing S(K) for some frame K:
#
# 1. Directly: S(K) = V(K) + ... + V(K+W-1)
#
# 2. From the preceding sum: S(K) = S(K-1) - V(K-1) + V(K+W-1)
#
# 3. From the succeeding sum: S(K) = S(K+1) - V(K+W) + V(K)
#
# The first option loads W frames and requires W-1 additions, whereas options
# two and three load just two frames and require one addition and one
# subtraction.  This suggests options two and three are vastly preferable for
# any window size larger than two.  On the other hand, options two and three can
# only be used if one of the neighboring sums have already been computed, so
# option one has to be used at least once.  Another consideration is that the
# options two and three create a serial data dependency on a previous frame,
# which hampers parallelism.  On a parallel system, it may be preferable to use
# option one more than once, since each resulting sum of frames can be used as a
# starting point for two parallel workers.
#
# To chose a suitable number of applications of option one, it makes sense to
# model the resulting performance.  We assume that each individual step is
# purely dominated by the cost of loading data, i.e., the cost of option one is
# W, whereas the cost of options two and three is 2.  If we assume a system with
# P perfectly parallel workers, the cost of executing option one reduces to W/P,
# and the cost of executing option two or three reduces to 2/min(2*M, P) where M
# is the number of uses of option one.  The number of uses of option two is
# consequently N-W-M+1.  The total cost is therefore:
#
# Cost(N, W, P, M) = (W/P)*M + (N-W-M+1)*2/min(2*M, P)


class Video_rolling_average(Video_change_precision):
    def rolling_average(self, window_size: int) -> Self:
        if window_size < 1:
            raise ValueError(f"Invalid window size: {window_size}")
        video = self
        video_size = len(video)
        if window_size > video_size:
            raise ValueError(
                f"Invalid window size {window_size} for video of length {video_size}."
            )
        bits = math.ceil(math.log2(window_size))
        # Ensure the intermediate sums of frames will not overflow.
        if video.precision + bits > MAX_PRECISION:
            video = video.change_precision(MAX_PRECISION - bits)
        # Choose the best algorithm for the given window size.
        return video._rolling_average_A(window_size, video.precision + bits)

    def _rolling_average_A(self, window_size: int, precision: int):
        """Works best for small window sizes."""
        array = da.overlap.map_overlap(
            _rolling_average_A,
            self._array,
            depth={0: (0, window_size)},
            trim=False,
            dtype=precision_dtype(precision),
            w=window_size,
        )
        return type(self)(array, precision)

    def differential_rolling_average(self, window_size: int) -> Self:
        return self  # TODO


def _rolling_average_A(x, w, block_info):
    """"""
    dtype = block_info[None]["dtype"]
    n = len(x)
    y = type(x)(x[0 : n - w + 1].shape, dtype=dtype)
    y[:] = 0
    for p in range(0, w):
        y += x[p : n - w + 1 + p]
    return y
