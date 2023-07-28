from __future__ import annotations

from piscat.video.actions import ChangePrecision
from piscat.video.baseclass import Precision, Video, precision_dtype
from piscat.video.map_batches import map_batches


class Video_change_precision(Video):
    def change_precision(self, precision: Precision):
        new_chunk_size = self.plan_chunk_size(self.shape, precision)
        (f, h, w) = self.shape
        return type(self)(
            list(
                map_batches(
                    self.batches(),
                    (new_chunk_size, h, w),
                    precision_dtype(precision),
                    lambda t, s: ChangePrecision(t, s, precision, self.precision),
                    count=f,
                )
            ),
            self.shape,
            precision,
        )
