from __future__ import annotations

import itertools
import os
import pathlib
import weakref
from typing import Callable, Iterable, Iterator, NamedTuple, Sequence, Union, overload

import filetype
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from piscat.io import FileReader
from piscat.io.ffmpeg import FFmpegReader, FFmpegWriter
from piscat.io.numpy import NumpyReader, NumpyWriter
from piscat.io.raw import RawReader, RawWriter

Array = np.ndarray

Path = Union[str, pathlib.Path]

BYTES_PER_CHUNK = 2**21


class VideoChunk:
    """
    The in-memory representation of several consecutive video frames.

    A video chunk can have two internal representations - either its data is
    stored directly as a NumPy array, or its data is expressed as the output of
    a particular video op.

    :param shape: A (frames, height, width) tuple describing the shape of the
        chunk.
    :param dtype: The NumPy dtype describing the elements of the chunk.
    :param data: A NumPy array containing the contents of the chunk.  Computed
        lazily.
    :param videos: A weak set of all videos referencing this chunk.
    :param users: A weak set of all video ops referencing this chunk.
    """

    _shape: tuple[int, int, int]
    _dtype: np.dtype
    _data: VideoOp | Array | None
    _videos: weakref.WeakSet[Video]
    _users: weakref.WeakSet[VideoOp]

    def __init__(
        self,
        shape: tuple[int, int, int],
        dtype: np.dtype,
        data: VideoOp | Array | None = None,
    ):
        if shape[0] == 0:
            data = Array(shape=shape, dtype=dtype)
        self._shape = shape
        self._dtype = dtype
        self._data = data
        self._videos = weakref.WeakSet()
        self._users = weakref.WeakSet()

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the video chunk as a (frames, height, width) tuple.
        """
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """
        Return the element type of each pixel of the video chunk.
        """
        return self._dtype

    @property
    def data(self) -> Array:
        """
        Return the NumPy array underlying the video chunk.

        Depending on the internal representation of the video chunk, this
        operation can be a simple reference to the existing array, or involve
        the evaluation of the entire compute graph whose root node is this video
        chunk.
        """
        if isinstance(self._data, Array):
            return self._data
        else:
            compute_chunks([self])
            assert isinstance(self._data, Array)
            assert self._data.shape == self._shape
            assert self._data.dtype == self._dtype
            return self._data

    @staticmethod
    def from_array(array: Array):
        shape = array.shape
        if len(shape) != 3:
            raise ValueError("Only rank three arrays can be converted to chunks.")
        return VideoChunk(shape=shape, dtype=array.dtype, data=array)

    def __repr__(self) -> str:
        return f"VideoChunk({self.shape!r}, {self.dtype!r}, id={id(self):#x})"

    def __len__(self) -> int:
        """
        Return the number of frames in the video chunk.
        """
        return self._shape[0]

    def __getitem__(self, index):
        """
        Return a selection of the video chunk's data.
        """
        return self.data[index]

    def __setitem__(self, index, value):
        """
        Fill a selection of the video chunk's data with the supplied value.
        """
        self.data[index] = value

    def __array__(self):
        return self.data


class Batch(NamedTuple):
    """
    A batch is a selection of some part of a video chunk.
    """

    chunk: VideoChunk
    start: int
    stop: int
    step: int = 1


Batches = Sequence[Batch]

Kernel = Callable[[Batches, Batches], None]


class VideoOp:
    """
    A video op describes how the contents of some video chunks can be computed,
    possibly depending on the contents of several other video chunks.

    :param targets: A list of batches that will be written to by this kernel's
        kernel.
    :param sources: A list of batches that will be read from by this kernel's
        kernel.
    :param kernel: The callable that will be invoked by this kernel once the
        data of one or more of its targets is accessed.
    :param args: Any extra arguments that have been supplied.
    :param kwargs: Any keyword arguments that have been supplied.
    """

    _targets: list[Batch]
    _sources: list[Batch]
    _kernel: Kernel
    _args: tuple
    _kwargs: dict

    def __init__(self, kernel: Kernel, targets: list[Batch], sources: list[Batch]):
        assert len(targets) > 0
        # Connect this kernel with its source and target chunks.
        self._targets = targets
        self._sources = sources
        self._kernel = kernel
        for target, _, _, _ in targets:
            assert not target._data
            target._data = self
        for source, _, _, _ in sources:
            source._users.add(self)

    @property
    def sources(self) -> list[Batch]:
        return self._sources

    @property
    def targets(self) -> list[Batch]:
        return self._targets

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    def run(self) -> None:
        # Allocate all targets.
        for target, _, _, _ in self.targets:
            assert target._data is not None
            if isinstance(target._data, VideoOp):
                target._data = Array(shape=target.shape, dtype=target.dtype)
            assert isinstance(target._data, Array)
        for source, _, _, _ in self.sources:
            assert isinstance(source._data, Array)
        # Run the kernel.
        self.kernel(self.targets, self.sources)
        # Mark each target as read-only.
        for target, _, _, _ in self.targets:
            target.data.setflags(write=False)
        # Sever the connection to each source.
        for source, _, _, _ in self.sources:
            assert isinstance(source._data, Array)
            source._users.remove(self)


def copy_kernel(targets: Batches, sources: Batches) -> None:
    assert len(targets) == 1
    (target, tstart, tstop, _) = targets[0]
    pos = tstart
    for source, sstart, sstop, _ in sources:
        count = sstop - sstart
        target[pos : pos + count] = source[sstart:sstop]
        pos += count
    assert pos == tstop


def get_fill_kernel(value) -> Kernel:
    def kernel(targets: Batches, sources: Batches):
        assert len(sources) == 0
        (target, tstart, tstop, tstep) = targets[0]
        target[tstart:tstop:tstep] = value

    return kernel


def get_reader_kernel(reader: FileReader, start: int, stop: int) -> Kernel:
    def kernel(targets: Batches, sources: Batches) -> None:
        assert len(targets) == 1
        assert len(sources) == 0
        (chunk, cstart, cstop, _) = targets[0]
        assert (stop - start) == (cstop - cstart)
        reader.read_chunk(chunk.data[cstart:cstop], start, stop)

    return kernel


def compute_chunks(chunks: Iterable[VideoChunk]) -> None:
    """
    Ensure that all the supplied video chunks and their dependencies have their
    data and metadata computed.
    """
    # Build a schedule.
    schedule: list[VideoOp] = []
    kernels: set[VideoOp] = set()

    def process(chunk: VideoChunk) -> None:
        data = chunk._data
        if data is None:
            raise RuntimeError("Encountered a chunk with undefined contents.")
        if isinstance(data, Array):
            return
        kernel = data
        if kernel not in kernels:
            kernels.add(kernel)
            for source, _, _, _ in kernel.sources:
                process(source)
            schedule.append(kernel)

    for video_chunk in chunks:
        process(video_chunk)
    # Execute the schedule.
    for kernel in tqdm(schedule, delay=0.5):
        kernel.run()


def chunks_from_batches(
    batches: Iterable[Batch],
    /,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    kernel: Kernel = copy_kernel,
    offset: int = 0,
    count: int | None = None,
) -> Iterator[VideoChunk]:
    """
    Process the data in all supplied batches and return it as an iterator over
    same-sized chunks.

    :param batches: An iterable over batches that supply all the data.
    :param shape: The shape of each resulting chunk.
    :param dtype: The type of each element of a resulting chunk.
    :param kernel: The function that reads data from one or more batches, and
        that writes its results to a single target batch that initializes one
        resulting chunk.
    :param offset: The number of frames of the first resulting chunk that are
        left uninitialized.
    :param count: The total number of target frames that will be defined, or
        None if this function should create chunks until the supplied batches
        are exhausted.

    :returns: An iterator over chunks of the supplied shape and dtype.
    """
    batches = iter(batches)
    if count == 0:
        raise StopIteration
    (f, h, w) = shape
    group: list[Batch] = []  # The batches that constitute the next chunk.
    gstart, gstop = 0, 0  # The interval of frames in the current group.
    ccount = 0  # The amount of chunks that have already been created.

    def merge_batches(batches: list[Batch], start: int, stop: int):
        chunk = VideoChunk(shape=shape, dtype=dtype)
        assert 0 <= start <= stop <= shape[0]
        VideoOp(kernel=kernel, targets=[Batch(chunk, start, stop)], sources=batches)
        return chunk

    while count is None or gstart < count:
        cstart = max(ccount * f, offset)
        cend = (ccount + 1) * f if not count else min((ccount + 1) * f, offset + count)
        clen = cend - cstart
        # Gather enough source chunks to fill one target chunk.
        while gstop - gstart < clen:
            try:
                batch = next(batches)
                assert batch.stop <= batch.chunk.shape[0]
            except StopIteration as e:
                if count:
                    raise ValueError("Not enough input chunks.") from e
                else:
                    if gstop - gstart > 0:
                        yield merge_batches(group, start=0, stop=gstop - gstart)
                    return
            (_, ch, cw) = batch.chunk.shape
            if ch != h or cw != w:
                raise ValueError("Cannot combine chunks with different frame sizes.")
            group.append(batch)
            gstop += batch.stop - batch.start
        rest = (gstop - gstart) - clen
        (last, lstart, lstop, lstep) = group[-1]
        group[-1] = Batch(last, lstart, lstop - rest, lstep)
        yield merge_batches(group, start=cstart - ccount * f, stop=cend - ccount * f)
        # Prepare for the next iteration.
        if rest == 0:
            group = []
        else:
            group = [Batch(last, lstop - rest, lstop, lstep)]
        gstart += clen
        ccount += 1


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
            chunk._videos.add(self)
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
    def __getitem__(self, index: slice) -> Video:
        ...

    @overload
    def __getitem__(self, index: tuple[int]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[slice]) -> Video:
        ...

    @overload
    def __getitem__(self, index: tuple) -> Array:
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
        chunks = chunks_from_batches(
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
            chunks_from_batches(
                [Batch(VideoChunk(shape=shape, dtype=dtype, data=array), 0, f)],
                shape=(chunk_size, h, w),
                dtype=array.dtype,
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
        for chunk, cstart, cstop, _ in self.batches():
            count = cstop - cstart
            writer.write_chunk(chunk.data[cstart:cstop], position)
            position += count
        return self


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)
