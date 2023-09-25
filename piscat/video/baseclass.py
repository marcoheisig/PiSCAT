from __future__ import annotations

from abc import ABC
from typing import Iterator

from piscat.video.actions import precision_dtype
from piscat.video.evaluation import Batch, Chunk, Dtype, batches_from_chunks

BYTES_PER_CHUNK = 2**21


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


class Video(ABC):
    """
    The abstract base class of all video related classes.

    This class is inherited from by each class that defines video methods.  In
    the end, one class in __init__.py combines the behavior of all these classes
    into one.

    :param shape: A tuple whose first element is the number of frames of the
        video, whose second element is the height of each frame, and whose third
        element is the width of each frame.
    :param precision: The number of bits of information in each of the video's
        pixels.
    :param chunks: A non-empty list of chunks that all have the same shape and
        dtype.
    :param chunk_offset: The position of the first frame of the video in the
        first chunk.  Elements of that chunk before this position are possibly
        uninitialized.
    :param chunk_shape: The shape of each of the video's chunks.
    :param chunk_dype: The dtype of each of the video's chunks.
    """

    _shape: tuple[int, int, int]
    _precision: int
    _chunks: list[Chunk]
    _chunk_offset: int

    def __init__(
        self,
        chunks: list[Chunk],
        shape: tuple[int, int, int],
        precision: int = 16,
        chunk_offset: int = 0,
    ):
        (f, h, w) = shape
        nchunks = len(chunks)
        chunk_size = 0 if nchunks == 0 else chunks[0].shape[0]
        chunk_shape = (chunk_size, h, w)
        dtype = precision_dtype(precision)
        for chunk in chunks:
            if not (chunk.shape == chunk_shape):
                raise ValueError("All chunks of a video must have the same shape.")
            if not (chunk.dtype == dtype):
                raise ValueError("All chunks of a video must have a suitable dtype.")
        if chunk_size > 0 and not 0 <= chunk_offset < chunk_size:
            raise ValueError("The chunk_offset argument must be within the first chunk.")
        if (nchunks * chunk_size - chunk_offset) < f:
            raise ValueError(f"Not enough chunks for a video of length {f}.")
        if (nchunks * chunk_size - chunk_offset - f) > chunk_size:
            raise ValueError(f"Too many chunks for a video of length {f}.")
        self._shape = shape
        self._precision = precision
        self._chunks = chunks
        self._chunk_offset = chunk_offset

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the video as a (frames, height, width) tuple.
        """
        return self._shape

    @property
    def dtype(self) -> Dtype:
        """
        Return the dtype of the video's array of pixels.
        """
        return precision_dtype(self.precision)

    @property
    def precision(self) -> int:
        """
        Return the number of bits of information per pixel of the video.
        """
        return self._precision

    @property
    def chunk_shape(self) -> tuple[int, int, int]:
        if len(self._chunks) == 0:
            return self._shape
        else:
            return self._chunks[0].shape

    @property
    def chunk_dtype(self) -> Dtype:
        return self._chunks[0].dtype

    def __len__(self):
        """
        Return the number of frames in the video.
        """
        return self.shape[0]

    @property
    def chunk_offset(self):
        return self._chunk_offset

    @property
    def chunk_size(self):
        return self.chunk_shape[0]

    @property
    def chunks(self):
        return self._chunks

    def __repr__(self):
        return f"<Video shape={self.shape!r} precision={self.precision!r} id={id(self):#x}>"

    def batches(self, start: int = 0, stop: int | None = None) -> Iterator[Batch]:
        if stop is None:
            stop = len(self)
        return batches_from_chunks(
            self.chunks, self.chunk_offset + start, self.chunk_offset + stop
        )

    @staticmethod
    def plan_chunk_size(shape: tuple[int, int, int], precision: int):
        """
        Return a reasonable chunk size for videos with the supplied shape and
        precision.

        The result is an integer between one and the constant BYTES_PER_CHUNK.
        """
        (f, h, w) = shape
        bytes_per_frame = h * w * ceildiv(precision, 8)
        return min(f, ceildiv(BYTES_PER_CHUNK, bytes_per_frame))
