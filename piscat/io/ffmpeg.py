from __future__ import annotations

import heapq
import pathlib
import subprocess
import sys
from typing import MutableSequence

import ffmpeg
import numpy as np

from piscat.io.fileio import FileReader, FileWriter


class FFmpegReader(FileReader):
    _duration: float
    _tmp_dtype: np.dtype
    _tmp_pix_fmt: str

    def __init__(self, path: pathlib.Path):
        if not path.exists or not path.is_file():
            raise ValueError(f"No video file at {path}.")
        metadata = ffmpeg.probe(str(path))
        stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        if not stream:
            raise ValueError(f"No video stream found in {path}.")
        f = int(stream.get("nb_frames", "1"))
        h = stream["height"]
        w = stream["width"]
        title: str = metadata.get("format", {}).get("tags", {}).get("title", "")
        dtype = np.dtype(title[13:]) if title.startswith("piscat_video_") else np.dtype("f4")
        self._duration = float(stream["duration"])
        (self._tmp_dtype, self._tmp_pix_fmt) = ffmpeg_dtype_and_pix_fmt(dtype)
        super().__init__(path, (f, h, w), dtype)

    def _read_chunk(self, array: MutableSequence, start: int, stop: int) -> None:
        (f, h, w) = self.shape
        duration = self._duration
        count = stop - start
        dt = duration / f
        out, _ = (
            ffmpeg.input(str(self.path), ss=dt * start, t=dt * count)
            .video.filter("format", self._tmp_pix_fmt)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt=self._tmp_pix_fmt,
                frames=count,
                s=f"{w}x{h}",
            )
            .run(capture_stdout=True, quiet=True)
        )
        data = np.frombuffer(out, self._tmp_dtype)
        data = data.reshape((count, w, h))
        data = np.transpose(data, (0, 2, 1))
        data = convert_data(data, self.dtype)
        array[0:count] = data


class FFmpegWriter(FileWriter):
    class DelayedWrite:
        start: int
        stop: int
        array: np.ndarray

        def __init__(self, start, stop, array):
            self.start = start
            self.stop = stop
            self.array = array

        def __lt__(self, other):
            return self.start < other.start

    _tmp_dtype: np.dtype
    _tmp_pix_fmt: str
    _heap: list[DelayedWrite] = []
    _process: subprocess.Popen | None = None
    _position: int = 0

    def __init__(self, path: pathlib.Path, shape: tuple[int, ...], dtype: np.dtype):
        super().__init__(path, shape, dtype)
        self._tmp_dtype, self._tmp_pix_fmt = ffmpeg_dtype_and_pix_fmt(self.dtype)

    def _initialize(self) -> None:
        super()._initialize()
        (_, h, w) = self.shape
        self._process = (
            ffmpeg.input("pipe:", format="rawvideo", s=f"{w}x{h}", pix_fmt=self._tmp_pix_fmt)
            .output(str(self.path), metadata=f"title=piscat_video_{self.dtype}")
            .run_async(pipe_stdin=True, quiet=True)
        )

    def _finalize(self) -> None:
        process = self._process
        assert process is not None
        assert process.stdin is not None
        process.stdin.close()
        process.wait()
        self._process = None
        super()._finalize()

    def _write_chunk(self, array: np.ndarray, position: int) -> None:
        process = self._process
        heap = self._heap
        array = np.transpose(convert_data(array, self._tmp_dtype), (0, 2, 1))
        count = len(array)
        if position == self._position:
            assert process is not None
            assert process.stdin is not None
            process.stdin.write(array.tobytes())
            self._position += count
            while len(heap) > 0:
                write = heap[0]
                if not write.start == self._position:
                    break
                heapq.heappop(heap)
                write.array.tofile(process.stdin)
                self._position += len(write.array)
        else:
            write = self.DelayedWrite(position, position + count, array)
            heapq.heappush(self._heap, write)


pix_fmt_u1 = "gray"
pix_fmt_u2 = "gray16le" if sys.byteorder == "little" else "gray16be"
pix_fmt_f4 = "grayf32le" if sys.byteorder == "little" else "grayf32be"


def ffmpeg_dtype_and_pix_fmt(dtype: np.dtype) -> tuple[np.dtype, str]:
    """
    Return the most suitable dtype and pixel format for decoding this video.
    """
    nbytes = dtype.itemsize
    if nbytes == 1:
        return (np.dtype("u1"), pix_fmt_u1)
    else:
        return (np.dtype("f4"), pix_fmt_f4)


def convert_data(data: np.ndarray, dtype: np.dtype):
    if data.dtype == dtype:
        return data
    elif data.dtype.kind == "u":
        if dtype.kind == "f":
            umax = 1 << (data.dtype.itemsize * 8)
            return data.astype(dtype) / np.array(umax, dtype)
        elif dtype.kind == "u":
            bits = (dtype.itemsize - data.dtype.itemsize) * 8
            if bits >= 0:
                return data.astype(dtype) << bits
            else:
                return (data >> -bits).astype(dtype)
        else:
            raise NotImplementedError()
    elif data.dtype.kind == "f":
        if dtype.kind == "f":
            return data.astype(dtype)
        elif dtype.kind == "u":
            umax = 1 << (dtype.itemsize * 8)
            return (data * umax).astype(dtype)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
