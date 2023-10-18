from __future__ import annotations

from typing import Iterable

import dask.array as da
from typing_extensions import Self

from piscat.video.change_precision import Video_change_precision


class Video_concatenate(Video_change_precision):
    @classmethod
    def concatenate(cls, videos: Iterable[Video_change_precision]) -> Self:
        videos = list(videos)
        if len(videos) == 0:
            raise ValueError("Cannot concatenate zero videos.")
        max_precision = max(video.precision for video in videos)
        _, height, width = videos[0].shape
        for video in videos[1:]:
            _, h, w = video.shape
            if (h, w) == (height, width):
                continue
            raise ValueError(f"Cannot concatenate {height}x{width} frames and {h}x{w} frames.")
        array = da.concatenate(video.change_precision(max_precision)._array for video in videos)
        return cls(array, max_precision)
