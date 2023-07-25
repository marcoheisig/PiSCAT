from __future__ import annotations

from typing_extensions import Self

from piscat.video.actions import Fill, dtype_decoder_and_precision
from piscat.video.baseclass import Video, ceildiv, precision_dtype
from piscat.video.evaluation import Array, Batch, Chunk


class Video_from_frame(Video):
    @classmethod
    def from_frame(cls, frame: Array, length: int = 1) -> Self:
        """
        Return a video whose frames are all equal to the supplied 2D array.
        """
        # First of all, create a copy of the frame so that later modifications
        # to it don't also modify parts of the resulting video.
        frame = frame.copy()
        if len(frame.shape) != 2:
            raise ValueError(f"Invalid frame: {frame}")
        h, w = frame.shape
        shape = (length, h, w)
        frame_dtype = frame.dtype
        decoder, precision = dtype_decoder_and_precision(frame_dtype)
        chunk_size = Video.plan_chunk_size(shape, precision)
        video_dtype = precision_dtype(precision)
        source = Batch(Chunk((1, h, w), frame_dtype, frame.reshape((1, h, w))), 0, 1)
        tmp = Batch(Chunk((1, h, w), video_dtype), 0, 1)
        decoder(tmp, source)
        cshape = (chunk_size, h, w)
        nchunks = ceildiv(length, chunk_size)
        chunks = [Chunk(cshape, video_dtype) for _ in range(nchunks)]
        for cn in range(nchunks):
            count = chunk_size if cn != nchunks - 1 else length - cn * chunk_size
            Fill(Batch(chunks[cn], 0, count), tmp)
        return cls(chunks, shape, precision)
