from __future__ import annotations

import numpy as np

from piscat.video.baseclass import Video
from piscat.video.evaluation import Array


class Video_to_array(Video):
    def to_array(self) -> Array:
        length = len(self)
        if length == 0:
            return self._chunks[0][0:0]
        chunks = self._chunks
        chunk_offset = self.chunk_offset
        nchunks = len(chunks)
        if nchunks == 1:
            return chunks[0][chunk_offset : chunk_offset + length]
        else:
            (chunk_size, h, w) = self.chunk_shape
            result = np.empty(shape=(length, h, w), dtype=self.dtype)
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
