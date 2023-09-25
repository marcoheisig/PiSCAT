from __future__ import annotations

from piscat.video.rolling_average import Video_rolling_average

from .concatenate import Video_concatenate
from .evaluation import Action, Batch, Chunk
from .from_array import Video_from_array
from .from_file import Video_from_file
from .from_frame import Video_from_frame
from .indexing import Video_indexing
from .to_array import Video_to_array
from .to_file import Video_to_file


class Video(
    Video_concatenate,
    Video_from_array,
    Video_from_file,
    Video_from_frame,
    Video_indexing,
    Video_rolling_average,
    Video_to_array,
    Video_to_file,
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


__all__ = ["Video", "Chunk", "Batch", "Action"]
