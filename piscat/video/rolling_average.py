from __future__ import annotations

import math

from typing_extensions import Self

from piscat.video.actions import UINT64, ChangePrecision, ForwardRollingAverage, Map, Sum
from piscat.video.baseclass import Video, precision_dtype
from piscat.video.evaluation import Batch, Chunk, batches_from_chunks


class Video_rolling_average(Video):
    def rolling_average(self, window_size: int) -> Self:
        (f, h, w) = self.shape
        if window_size < 1:
            raise RuntimeError(f"Invalid window size: {window_size}")
        if window_size > f:
            raise RuntimeError("Not enough video frames for computing the rolling average.")
        shape = (f - window_size + 1, h, w)
        precision = min(self.precision + math.ceil(math.log2(window_size)), 32)
        extra_bits = self.precision - precision
        factor = 2**extra_bits
        chunk_size = self.plan_chunk_size(shape, precision)
        acc: Batch = Batch(Chunk((1, *self.shape[1:]), UINT64), 0, 1)
        Sum(acc, list(self.batches(0, window_size)))

        def FRA(target, left, right):
            nonlocal acc
            new_acc = Batch(Chunk(acc.chunk.shape, UINT64), 0, 1)
            ForwardRollingAverage(target, new_acc, left, right, acc, window_size, factor)
            acc = new_acc

        chunk_shape = (chunk_size, h, w)
        dtype = precision_dtype(precision)
        chunks = []
        for _ in range(0, shape[0], chunk_size):
            chunks.append(Chunk(chunk_shape, dtype))
        Map(
            FRA,
            batches_from_chunks(chunks, 0, f - window_size),
            self.batches(0, f - window_size),
            self.batches(window_size, f),
        )
        last = next(batches_from_chunks(chunks, f - window_size, f - window_size + 1))
        ChangePrecision(last, acc, precision, 64)
        return type(self)(chunks, shape, precision)

    def differential_rolling_average(self, window_size: int) -> Self:
        return self  # TODO
