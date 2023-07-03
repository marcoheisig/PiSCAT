from __future__ import annotations

from typing_extensions import Self

from piscat.video.baseclass import Video
from piscat.video.evaluation import Array, Batch, Batches, Kernel, VideoChunk, VideoOp, ceildiv


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
        dtype = frame.dtype
        csize = Video.plan_chunk_size(shape=(length, h, w), dtype=dtype)
        cshape = (csize, h, w)
        nchunks = ceildiv(length, csize)
        chunks = [VideoChunk(cshape, dtype) for _ in range(nchunks)]
        for cn in range(nchunks - 1):
            VideoOp(get_fill_kernel(frame), [Batch(chunks[cn], 0, csize)], [])
        if nchunks > 0:
            cn = nchunks - 1
            VideoOp(get_fill_kernel(frame), [Batch(chunks[cn], 0, length - cn * csize)], [])
        return cls(chunks, 0, length)


def get_fill_kernel(value) -> Kernel:
    def kernel(targets: Batches, sources: Batches):
        assert len(sources) == 0
        (target, tstart, tstop) = targets[0]
        target[tstart:tstop] = value

    return kernel
