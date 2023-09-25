from __future__ import annotations

import numpy as np

from piscat.video.actions import precision_next_power_of_two
from piscat.video.change_precision import Video_change_precision
from piscat.video.evaluation import Array, compute_chunks


class Video_to_array(Video_change_precision):
    def to_array(self) -> Array:
        video = self.change_precision(precision_next_power_of_two(self.precision))
        length = len(video)
        chunks = video._chunks
        chunk_offset = video.chunk_offset
        nchunks = len(chunks)
        if nchunks == 1:
            return chunks[0][chunk_offset : chunk_offset + length]
        else:
            compute_chunks(chunks)
            (chunk_size, h, w) = video.chunk_shape
            result = np.empty(shape=(length, h, w), dtype=video.dtype)
            pos = 0
            for cn in range(nchunks):
                chunk = chunks[cn]
                start = chunk_offset if cn == 0 else 0
                stop = (
                    (chunk_offset + length) - cn * chunk_size
                    if cn == (nchunks - 1)
                    else chunk_size
                )
                count = stop - start
                result[pos : pos + count] = chunk[start:stop]
                pos += count
            return result

    def __array__(self) -> Array:
        return self.to_array()
