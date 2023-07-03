from __future__ import annotations

from typing_extensions import Self

from piscat.video.baseclass import Video
from piscat.video.evaluation import Array, Batch, VideoChunk, copy_kernel
from piscat.video.map_batches import map_batches


class Video_from_array(Video):
    @classmethod
    def from_array(cls, array: Array, /, chunk_size: int | None = None) -> Self:
        """
        Return a video whose contents are the same as the supplied array.
        """
        shape = array.shape
        dtype = array.dtype
        if len(shape) != 3:
            raise ValueError("Can only turn three-dimensional arrays into videos.")
        (f, h, w) = shape
        if f == 0:
            return cls([VideoChunk(shape=shape, dtype=dtype)], 0, 0)
        if chunk_size is None:
            chunk_size = Video.plan_chunk_size(shape=shape, dtype=dtype)
        chunks = list(
            map_batches(
                [Batch(VideoChunk(shape=shape, dtype=dtype, data=array), 0, f)],
                shape=(chunk_size, h, w),
                dtype=array.dtype,
                kernel=copy_kernel,
            )
        )
        return cls(chunks, 0, f)
