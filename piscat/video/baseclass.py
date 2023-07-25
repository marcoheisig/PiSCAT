from __future__ import annotations

from abc import ABC
from typing import Iterator, Literal, Union

import numpy as np

from piscat.video.evaluation import Batch, Chunk, Dtype

Precision = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
]

ExtendedPrecision = Union[Precision, Literal[25, 26, 27, 28, 29, 30, 31, 32]]

BYTES_PER_CHUNK = 2**21


def precision_dtype(precision: Precision) -> Dtype:
    if precision <= 8:
        return np.dtype(np.uint8)
    elif precision <= 16:
        return np.dtype(np.uint16)
    elif precision <= 24:
        return np.dtype(np.uint32)
    else:
        raise ValueError(f"Invalid video precision: {precision}")


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
    _precision: Precision
    _chunks: list[Chunk]
    _chunk_offset: int

    def __init__(
        self,
        chunks: list[Chunk],
        shape: tuple[int, int, int],
        precision: Precision = 16,
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
    def precision(self) -> Precision:
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
        stop = len(self) if stop is None else stop
        length = stop - start
        if length == 0:
            return
        chunks = self._chunks
        chunk_size = self.chunk_size
        chunk_offset = self.chunk_offset
        pstart = start + chunk_offset
        plast = stop + chunk_offset - 1
        cstart = pstart // chunk_size
        clast = plast // chunk_size
        for cn in range(cstart, clast + 1):
            yield Batch(
                chunks[cn],
                max(0, pstart - cn * chunk_size),
                min(chunk_size, (plast + 1) - cn * chunk_size),
            )

    @staticmethod
    def plan_chunk_size(shape: tuple[int, int, int], precision: Precision):
        """
        Return a reasonable chunk size for videos with the supplied shape and
        precision.

        The result is an integer between one and the constant BYTES_PER_CHUNK.
        """
        (f, h, w) = shape
        bytes_per_frame = h * w * ceildiv(precision, 8)
        return min(f, ceildiv(BYTES_PER_CHUNK, bytes_per_frame))


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)
