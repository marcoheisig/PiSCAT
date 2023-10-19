from __future__ import annotations

from piscat.video.change_precision import Video_change_precision
from piscat.video.concatenate import Video_concatenate
from piscat.video.from_array import Video_from_array
from piscat.video.from_raw_file import Video_from_raw_file
from piscat.video.indexing import Video_indexing
from piscat.video.rolling_average import Video_rolling_average
from piscat.video.to_array import Video_to_array
from piscat.video.to_raw_file import Video_to_raw_file


class Video(
    Video_to_raw_file,
    Video_to_array,
    Video_indexing,
    Video_rolling_average,
    Video_from_raw_file,
    Video_concatenate,
    Video_change_precision,
    Video_from_array,
):
    """
    An efficient collection of video frames.

    Each video is represented as a list of chunks of the same size.  Chunks are
    evaluated lazily, so you only pay for video frames that are actually
    accessed.  All video processing algorithms operate one chunk at a time, to
    improve memory locality and conserve main memory.

    :param shape: A (frames, height, width) tuple describing the extent of the
        video.
    :param dtype: The NumPy dtype describing each pixel of the video.
    :param data: The NumPy array containing all frames of the video.  Accessing
        a video's data is expensive, because the video frames are normally
        stored in a more efficient manner and the resulting NumPy array has to
        be allocated and initialized each time the data is accessed.
    :param chunk_shape: A (frames, height, width) tuple describing the size of
        the chunks into which the video is partitioned internally.
    :param chunk_offset: The position of the video's first frame in the first
        chunk.
    """

    ...


__all__ = ["Video"]
