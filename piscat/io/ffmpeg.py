from __future__ import annotations

import heapq
import pathlib
import subprocess
import sys
from typing import MutableSequence

import ffmpeg
import numpy as np

from piscat.io.fileio import FileReader, FileWriter

pix_fmts_8bit = {
    "gray",
    "monow",
    "monob",
    "pal8",
    "bgr8",
    "bgr4",
    "bgr4_byte",
    "rgb8",
    "rgb4",
    "rgb4_byte",
    "vaapi_moco",
    "vaapi_idct",
    "vaapi_vld",
    "dxva2_vld",
    "vdpau",
    "qsv",
    "mmal",
    "d3d11va_vld",
    "cuda",
    "bayer_bggr8",
    "bayer_rggb8",
    "bayer_gbrg8",
    "bayer_grbg8",
    "xvmc",
    "videotoolbox_vld",
    "mediacodec",
    "d3d11",
    "drm_prime",
    "opencl",
}

pix_fmts_16bit = {
    "yuv420p",
    "yuyv422",
    "yuv422p",
    "yuv410p",
    "yuv411p",
    "yuvj420p",
    "yuvj422p",
    "uyvy422",
    "uyyvyy411",
    "nv12",
    "nv21",
    "gray16be",
    "gray16le",
    "yuv440p",
    "yuvj440p",
    "rgb565be",
    "rgb565le",
    "rgb555be",
    "rgb555le",
    "bgr565be",
    "bgr565le",
    "bgr555be",
    "bgr555le",
    "rgb444le",
    "rgb444be",
    "bgr444le",
    "bgr444be",
    "ya8",
    "yuv420p9be",
    "yuv420p9le",
    "yuv420p10be",
    "yuv420p10le",
    "nv16",
    "yvyu422",
    "yuvj411p",
    "bayer_bggr16le",
    "bayer_bggr16be",
    "bayer_rggb16le",
    "bayer_rggb16be",
    "bayer_gbrg16le",
    "bayer_gbrg16be",
    "bayer_grbg16le",
    "bayer_grbg16be",
    "p010le",
    "p010be",
    "gray12be",
    "gray12le",
    "gray10be",
    "gray10le",
    "gray9be",
    "gray9le",
    "gray14be",
    "gray14le",
}

pix_fmt_u1 = "gray"
pix_fmt_u2 = "gray16le" if sys.byteorder == "little" else "gray16be"
pix_fmt_f4 = "grayf32le" if sys.byteorder == "little" else "grayf32be"


class FFmpegReader(FileReader):
    _duration: float
    _format: str
    _output_pix_fmt: str

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
        self._duration = float(stream["duration"])
        (dtype, output_pix_fmt) = dtype_and_output_pix_fmt(stream["pix_fmt"])
        self._output_pix_fmt = output_pix_fmt
        super().__init__(path, (f, h, w), dtype)

    def _read_chunk(self, array: MutableSequence, start: int, stop: int) -> None:
        (f, h, w) = self.shape
        duration = self._duration
        dtype = self.dtype
        count = stop - start
        dt = duration / f
        out, _ = (
            ffmpeg.input(str(self.path), ss=dt * start, t=dt * count)
            .video.filter("format", "gray")
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt=self._output_pix_fmt,
                frames=count,
                s=f"{w}x{h}",
            )
            .run(capture_stdout=True, quiet=True)
        )
        data = np.frombuffer(out, dtype).reshape((count, w, h))
        array[0:count] = np.transpose(data, (0, 2, 1))


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

    _output_dtype: np.dtype
    _output_pix_fmt: str
    _heap: list[DelayedWrite] = []
    _process: subprocess.Popen | None = None
    _position: int = 0

    def __init__(self, path: pathlib.Path, shape: tuple[int, ...], dtype: np.dtype):
        super().__init__(path, shape, dtype)
        nbytes = dtype.itemsize
        if nbytes <= 8:
            self._output_dtype = np.dtype("u1")
            self._output_pix_fmt = pix_fmt_u1
        elif nbytes <= 16:
            self._output_dtype = np.dtype("u2")
            self._output_pix_fmt = pix_fmt_u2
        else:
            self._output_dtype = np.dtype("f4")
            self._output_pix_fmt = pix_fmt_f4

    def _initialize(self) -> None:
        super()._initialize()
        (_, h, w) = self.shape
        self._process = (
            ffmpeg.input("pipe:", format="rawvideo", s=f"{w}x{h}", pix_fmt=self._output_pix_fmt)
            .output(str(self.path))
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
        array = np.transpose(np.array(array, dtype=self._output_dtype), (0, 2, 1))
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


def dtype_and_output_pix_fmt(pix_fmt) -> tuple[np.dtype, str]:
    """
    Return the most suitable dtype and pixel format for decoding this video.
    """
    if pix_fmt in pix_fmts_8bit:
        return (np.dtype("u1"), pix_fmt_u1)
    elif pix_fmt in pix_fmts_16bit:
        return (np.dtype("u2"), pix_fmt_u2)
    else:
        return (np.dtype("f4"), pix_fmt_f4)
