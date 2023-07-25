from __future__ import annotations

import itertools
from typing import Iterable

from typing_extensions import Self

from piscat.video.actions import Copy
from piscat.video.baseclass import Video, precision_dtype
from piscat.video.map_batches import map_batches


class Video_concatenate(Video):
    @classmethod
    def concatenate(cls, videos: Iterable[Video]) -> Self:
        videos = list(videos)
        if len(videos) == 0:
            raise ValueError("Cannot concatenate zero videos.")
        precision = videos[0].precision
        length, height, width = videos[0].shape
        for video in videos[1:]:
            (f, h, w) = video.shape
            if (h, w) != (height, width):
                raise ValueError(
                    f"Cannot concatenate {height}x{width} frames and {h}x{w} frames."
                )
            # TODO allow concatenating videos with varying precision.
            if video.precision != precision:
                raise ValueError("Cannot concatenate videos with varying precision.")
            length += f
        shape = (length, height, width)
        if length == 0:
            return cls([], shape, precision=precision)
        chunk_shape = (Video.plan_chunk_size(shape, precision), height, width)
        batches = itertools.chain.from_iterable(video.batches() for video in videos)
        dtype = precision_dtype(precision)
        chunks = list(map_batches(batches, chunk_shape, dtype, Copy, count=length))
        return cls(chunks, shape, chunk_offset=0, precision=precision)
