from __future__ import annotations

import pathlib
from typing import Union

import filetype
from typing_extensions import Self

from piscat.io import FileReader
from piscat.io.ffmpeg import FFmpegReader
from piscat.io.numpy import NumpyReader
from piscat.video.from_raw_file import Video_from_raw_file

Path = Union[str, pathlib.Path]


class Video_from_file(Video_from_raw_file):
    @classmethod
    def from_file(cls, path: Path) -> Self:
        """
        Return a video whose contents are read in from a file.

        :param path: The path to the video to read.
        :param format: The format of the video.  Valid options are "raw", "tif",
            "fits", "fli", "h5", "npy", and any format supported by ffmpeg such
            as "avi", "webm", and "mp4".  If no format is specified, an attempt
            is made to infer it from the suffix of the supplied path, or by
            looking at the file itself.
        :param shape: The expected shape of the video being read in.  If
            supplied, an error is raised if the video's shape differs from its
            expectation.
        :param dtype: The element type of the resulting video.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"No such file: {path}")
        # Determine the file extension.
        if path.suffix and path.suffix.startswith("."):
            extension = path.suffix[1:]
        else:
            extension = filetype.guess_extension(path)
        # Create an appropriate video reader.
        reader: FileReader
        if extension == "raw":
            raise RuntimeError("Please use Video.from_raw_file to load raw videos.")
        elif extension == "npy":
            reader = NumpyReader(path)
        elif extension == "fits":
            raise NotImplementedError()
        elif extension == "fli":
            raise NotImplementedError()
        elif extension == "h5":
            raise NotImplementedError()
        else:
            reader = FFmpegReader(path)
        # Create the video.
        raise NotImplementedError()
