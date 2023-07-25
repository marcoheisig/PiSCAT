from __future__ import annotations

from typing_extensions import Self

from piscat.video.actions import dtype_decoder_and_precision
from piscat.video.baseclass import Video, precision_dtype
from piscat.video.evaluation import Array, Batch, Chunk
from piscat.video.map_batches import map_batches


class Video_from_array(Video):
    @classmethod
    def from_array(cls, array: Array) -> Self:
        """
        Return a video whose contents are derived from the supplied array.
        """
        shape = array.shape
        if len(shape) != 3:
            raise ValueError("Can only turn three-dimensional arrays into videos.")
        (f, h, w) = shape
        decoder, precision = dtype_decoder_and_precision(array.dtype)
        # Videos with zero frames don't require lazy evaluation.
        if f == 0:
            return cls([], shape)
        chunk_size = Video.plan_chunk_size(shape=shape, precision=precision)
        chunks = list(
            map_batches(
                [Batch(Chunk(shape, array.dtype, array), 0, f)],
                (chunk_size, h, w),
                dtype=precision_dtype(precision),
                action=decoder,
            )
        )
        return cls(chunks, shape, precision=precision)
