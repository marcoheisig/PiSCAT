from __future__ import annotations

import numpy.typing as npt

from piscat.video.to_array import Video_to_array, array_from_video_data
from piscat.video.utilities import Filename, path_to_new_file


class Video_to_raw_file(Video_to_array):
    def to_raw_file(
        self, filename: Filename, dtype: npt.DTypeLike | None = None, overwrite: bool = False
    ) -> None:
        path = path_to_new_file(filename, overwrite=overwrite)
        dtype = self.dtype if dtype is None else dtype
        array_from_video_data(self._array, self.precision, dtype).compute().tofile(path)
