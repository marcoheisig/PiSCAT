from __future__ import annotations

import itertools
from typing import Iterable

from typing_extensions import Self

from piscat.video.baseclass import Video
from piscat.video.evaluation import VideoChunk, copy_kernel
from piscat.video.map_batches import map_batches


class Video_concatenate(Video):
    @classmethod
    def concatenate(cls, videos: Iterable[Video]) -> Self:
        videos = list(videos)
        if len(videos) == 0:
            raise ValueError("Cannot concatenate zero videos.")
        dtype = videos[0].dtype
        length, height, width = videos[0].shape
        for video in videos[1:]:
            (f, h, w) = video.shape
            if (h, w) != (height, width):
                raise ValueError(
                    f"Cannot concatenate {height}x{width} frames and {h}x{w} frames."
                )
            if video.dtype != dtype:
                raise ValueError("Cannot concatenate videos with different element types.")
            length += f
        shape = (length, height, width)
        if length == 0:
            return cls([VideoChunk(shape=shape, dtype=dtype)], 0, 0)
        chunk_shape = (Video.plan_chunk_size(shape=shape, dtype=dtype), height, width)
        batches = itertools.chain.from_iterable(video.batches() for video in videos)
        chunks = map_batches(
            batches, shape=chunk_shape, dtype=dtype, kernel=copy_kernel, count=length
        )
        return cls(chunks, 0, length)
