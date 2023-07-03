from __future__ import annotations

from abc import ABC
from typing import Iterable, Iterator

import numpy as np
import numpy.typing as npt

from piscat.video.evaluation import BYTES_PER_CHUNK, Batch, VideoChunk, ceildiv


class Video(ABC):
    """
    The abstract base class of all video related classes.

    This class is inherited from by each class that defines video methods.  In
    the end, one class in __init__.py combines the behavior of all these
    classes into one.
    """

    # A non-empty list of chunks that all have the same shape and dtype.
    _chunks: list[VideoChunk]
    # The shape of the video, which is a tuple whose first element is the number
    # of frames of the video, whose second element is the height of each, and
    # whose third element is the width of each frame.
    _shape: tuple[int, int, int]
    # The position of the first frame of the video in the first chunk.  Elements
    # of that chunk before this position are possibly uninitialized.
    _chunk_offset: int

    def __init__(
        self, chunks: Iterable[VideoChunk], chunk_offset: int = 0, length: int | None = None
    ):
        chunks = list(chunks)
        nchunks = len(chunks)
        if nchunks == 0:
            raise ValueError("Cannot create a video with zero chunks.")
        chunk0_shape = chunks[0].shape
        chunk0_dtype = chunks[0].dtype
        for chunk in chunks[1:nchunks]:
            if not (chunk.shape == chunk0_shape):
                raise ValueError("All chunks of a video must have the same size.")
            if not (chunk.dtype == chunk0_dtype):
                raise ValueError("All chunks of a video must have the same dtype.")
        (chunk_size, h, w) = chunk0_shape
        nframes = chunk_size * len(chunks)
        length = (nframes - chunk_offset) if length is None else length
        if (chunk_offset + length) > nframes:
            raise ValueError("Not enough chunks.")
        if chunk_size > 0 and not 0 <= chunk_offset < chunk_size:
            raise ValueError("The chunk_offset argument must be within the first chunk.")
        if chunk_size > 0 and not (chunk_offset + length) > nframes - chunk_size:
            raise ValueError("Too many chunks.")
        self._chunks = chunks
        self._shape = (length, h, w)
        self._chunk_offset = chunk_offset

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the video as a (frames, height, width) tuple.
        """
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._chunks[0].dtype

    @property
    def chunk_shape(self):
        return self._chunks[0].shape

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

    def __repr__(self):
        return f"<Video shape={self.shape!r} dtype={self.dtype!r} id={id(self):#x}>"

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
    def plan_chunk_size(shape: tuple[int, int, int], dtype: npt.DTypeLike):
        """
        Return a reasonable chunk size for videos with the supplied height,
        width, and dtype.
        """
        (f, h, w) = shape
        return min(f, ceildiv(BYTES_PER_CHUNK, h * w * np.dtype(dtype).itemsize))
