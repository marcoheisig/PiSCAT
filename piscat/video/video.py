from __future__ import annotations

import itertools
import os
import pathlib
from typing import Iterable, Iterator, Union, overload

import filetype
import numpy as np
import numpy.typing as npt

from piscat.io import FileReader, FileWriter
from piscat.io.ffmpeg import FFmpegReader, FFmpegWriter
from piscat.io.numpy import NumpyReader, NumpyWriter
from piscat.io.raw import RawReader, RawWriter
from piscat.video.evaluation import (
    BYTES_PER_CHUNK,
    Array,
    Batch,
    Batches,
    Kernel,
    VideoChunk,
    VideoOp,
    copy_kernel,
)
from piscat.video.patterns import map_batches

Path = Union[str, pathlib.Path]

EllipsisType = type(Ellipsis)

Slice = Union[slice, EllipsisType]


class Video:
    """
    An efficient container for consecutive 2D images and metadata.

    Internally, each video is represented as a list of chunks of the same size.
    Chunks are evaluated lazily, so you only pay for video frames that are
    actually accessed.  All video processing algorithms operate one chunk at a
    time, to improve memory locality.

    :param shape: A (frames, height, width) tuple describing the extent of the
        video.
    :param dtype: The NumPy dtype describing each pixel of the video.
    :param data: The NumPy array containing all frames of the video.  Accessing
        a video's data is expensive, because the video frames are normally
        stored in a more efficient manner and the resulting NumPy array has to
        be allocated and initialized each time the data is accessed.
    :param chunk_shape: A (frames, height, width) tuple describing the size of
        the chunks into which the video is partitioned internally.
    :param chunk_offset: The position of the video's first frame in the first chunk.
    """

    _chunks: list[VideoChunk]
    _shape: tuple[int, int, int]
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
        for chunk in chunks:
            chunk._users.add(self)
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

    def __array__(self):
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

    @overload
    def __getitem__(self, index: int) -> Array:
        ...

    @overload
    def __getitem__(self, index: Slice) -> Video:
        ...

    @overload
    def __getitem__(self, index: tuple[int]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int, int]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[Slice, ...]) -> Video:
        ...

    def __getitem__(self, index):
        """
        Return a particular part of the video.
        """
        size = self.shape[0]
        (csize, h, w) = self.chunk_shape
        chunks = self._chunks

        def icheck(index):
            if not (0 <= index < size):
                raise IndexError(f"Invalid index {index} for a video with {size} frames.")

        if isinstance(index, int):
            icheck(index)
            p = self.chunk_offset + index
            chunk = chunks[p // csize]
            return chunk[p % csize]
        elif isinstance(index, slice):
            istart = 0 if index.start is None else index.start
            istop = len(self) if index.stop is None else max(istart, index.stop)
            istep = 1 if index.step is None else index.step
            assert istep == 1  # TODO
            if istart == istop:
                return Video.from_array(Array(shape=(0, h, w), dtype=self.dtype))
            if istart == 0 and istop == size:
                return self
            icheck(istart)
            count = (istop - istart) // istep
            ilast = max(istart, istart + count - 1)
            icheck(ilast)
            pstart = self.chunk_offset + istart
            plast = self.chunk_offset + ilast
            return Video(chunks[pstart // csize : (plast // csize) + 1], pstart % csize, count)
        elif isinstance(index, tuple):
            i0 = index[0]
            if isinstance(i0, int):
                return self[i0].__getitem__(index[1:])
            else:
                # TODO
                return np.array(self)[index]
        else:
            raise ValueError(f"Invalid video index: {index}")

    def __iter__(self) -> Iterator[Array]:
        for index in range(len(self)):
            yield self[index]

    def batches(self) -> Iterator[Batch]:
        length = len(self)
        chunks = self._chunks
        nchunks = len(chunks)
        chunk_size = self.chunk_shape[0]
        chunk_offset = self.chunk_offset
        yield Batch(chunks[0], chunk_offset, min(chunk_size, chunk_offset + length))
        for cn in range(1, nchunks):
            start = 0
            stop = min(chunk_size, chunk_offset + length - cn * chunk_size)
            yield Batch(chunks[cn], start, stop)

    @staticmethod
    def concatenate(videos: Iterable[Video]):
        videos = list(videos)
        if len(videos) == 0:
            raise ValueError("Cannot concatenate zero videos.")
        dtype = videos[0].dtype
        length, height, width = videos[0].shape
        for video in videos[1:]:
            (f, h, w) = video.shape
            if (h, w) != (height, width):
                raise ValueError(
                    f"Cannot concatenate {height}x{width} frames and {h}x{w} frames."
                )
            if video.dtype != dtype:
                raise ValueError("Cannot concatenate videos with different element types.")
            length += f
        shape = (length, height, width)
        if length == 0:
            return Video([VideoChunk(shape=shape, dtype=dtype)], 0, 0)
        chunk_shape = (Video.plan_chunk_size(shape=shape, dtype=dtype), height, width)
        batches = itertools.chain.from_iterable(video.batches() for video in videos)
        chunks = map_batches(
            batches, shape=chunk_shape, dtype=dtype, kernel=copy_kernel, count=length
        )
        return Video(chunks, 0, length)

    @staticmethod
    def plan_chunk_size(shape: tuple[int, int, int], dtype: npt.DTypeLike):
        """
        Return a reasonable chunk size for videos with the supplied height,
        width, and dtype.
        """
        (f, h, w) = shape
        return min(f, ceildiv(BYTES_PER_CHUNK, h * w * np.dtype(dtype).itemsize))

    @staticmethod
    def from_array(array: Array) -> Video:
        """
        Return a video whose contents are the same as the supplied array.
        """
        shape = array.shape
        dtype = array.dtype
        if len(shape) != 3:
            raise ValueError("Can only turn three-dimensional arrays into videos.")
        (f, h, w) = shape
        if f == 0:
            return Video([VideoChunk(shape=shape, dtype=dtype)], 0, 0)
        chunk_size = Video.plan_chunk_size(shape=shape, dtype=dtype)
        chunks = list(
            map_batches(
                [Batch(VideoChunk(shape=shape, dtype=dtype, data=array), 0, f)],
                shape=(chunk_size, h, w),
                dtype=array.dtype,
                kernel=copy_kernel,
            )
        )
        return Video(chunks, 0, f)

    @staticmethod
    def from_frame(frame: Array, length: int = 1) -> Video:
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
        return Video(chunks, 0, length)

    @staticmethod
    def from_file(path: Path, /, chunk_size: int | None = None) -> Video:
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
        shape = reader.shape
        dtype = reader.dtype
        if chunk_size is None:
            chunk_size = Video.plan_chunk_size(shape, dtype)
        (f, h, w) = shape
        chunks = []
        for start in range(0, f, chunk_size):
            stop = min(f, start + chunk_size)
            count = stop - start
            chunk = VideoChunk((chunk_size, h, w), dtype)
            kernel = get_reader_kernel(reader, start, stop)
            VideoOp(kernel, [Batch(chunk, 0, count)], [])
            chunks.append(chunk)
        return Video(chunks, length=shape[0])

    @staticmethod
    def from_raw_file(
        path: Path,
        shape: tuple[int, int, int],
        dtype=npt.DTypeLike,
        /,
        chunk_size: int | None = None,
    ) -> Video:
        if isinstance(path, str):
            path = pathlib.Path(path)
        dtype = np.dtype(dtype)
        reader = RawReader(path, shape, np.dtype(dtype))
        if chunk_size is None:
            chunk_size = Video.plan_chunk_size(shape, dtype)
        (f, h, w) = shape
        chunks = []
        for start in range(0, f, chunk_size):
            stop = min(f, start + chunk_size)
            count = stop - start
            chunk = VideoChunk((chunk_size, h, w), dtype)
            kernel = get_reader_kernel(reader, start, stop)
            VideoOp(kernel, [Batch(chunk, 0, count)], [])
            chunks.append(chunk)
        return Video(chunks, length=shape[0])

    def to_file(self, path: Path, /, overwrite: bool = False) -> Video:
        if isinstance(path, str):
            path = pathlib.Path(path)
        if path.exists():
            if overwrite:
                os.remove(path)
            else:
                raise ValueError(f"The file named {path} already exists.")
        suffix = path.suffix
        extension = suffix[1:] if suffix.startswith(".") else "npy"
        writer: FileWriter
        if extension == "":
            raise ValueError(f"Couldn't determine the type of {path} (missing suffix).")
        if extension == "raw":
            writer = RawWriter(path, self.shape, self.dtype)
        elif extension == "npy":
            writer = NumpyWriter(path, self.shape, self.dtype)
        elif extension == "fits":
            raise NotImplementedError()
        elif extension == "fli":
            raise NotImplementedError()
        elif extension == "h5":
            raise NotImplementedError()
        else:
            writer = FFmpegWriter(path, self.shape, self.dtype)
        position = 0
        for chunk, cstart, cstop in self.batches():
            count = cstop - cstart
            writer.write_chunk(chunk.data[cstart:cstop], position)
            position += count
        return self


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def get_fill_kernel(value) -> Kernel:
    def kernel(targets: Batches, sources: Batches):
        assert len(sources) == 0
        (target, tstart, tstop) = targets[0]
        target[tstart:tstop] = value

    return kernel


def get_reader_kernel(reader: FileReader, start: int, stop: int) -> Kernel:
    def kernel(targets: Batches, sources: Batches) -> None:
        assert len(targets) == 1
        assert len(sources) == 0
        (chunk, cstart, cstop) = targets[0]
        assert (stop - start) == (cstop - cstart)
        reader.read_chunk(chunk.data[cstart:cstop], start, stop)

    return kernel
