from multiprocessing import Semaphore, Value
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from core.dtypes import BufferSize, FrameBuffer, FrameShape, IndexBuffer, SemaphoreType


class SharedBuffer:
    def __init__(self, frame_shape: FrameShape, buffer_size: BufferSize, shm_name=None):
        self.buffer_size = buffer_size
        self.frame_shape = frame_shape
        self.frame_size = np.prod(frame_shape) * np.dtype(np.uint8).itemsize

        # Attach to existing SHM
        if shm_name:
            self.shm = SharedMemory(name=shm_name)
            self.shm_index = SharedMemory(name=f"{shm_name}_index")
        else:
            self.shm = SharedMemory(
                create=True, size=self.buffer_size * self.frame_size
            )
            self.shm_index = SharedMemory(
                create=True, size=self.buffer_size * np.dtype(np.int32).itemsize
            )

        # Raw buffer (im_shape.flatten() * buffer_size
        self.buffer: FrameBuffer = np.ndarray(
            (self.buffer_size, *self.frame_shape), dtype=np.uint8, buffer=self.shm.buf
        )
        # Index buffer (buffer_size)
        self.index_buffer: IndexBuffer = np.ndarray(
            (self.buffer_size,), dtype=np.int32, buffer=self.shm_index.buf
        )

        if not shm_name:
            self.index_buffer.fill(-1)  # Initialize with no valid frames

        # Semaphore for a new frame
        self.frame_available = Semaphore(0)
        # Semaphore for a new empty space in buffer to write frame
        self.space_available = Semaphore(buffer_size)
        self.latest_index = Value("i", -1)  # Start at -1, no frames written yet

    def cleanup(self):
        self.shm.close()
        self.shm.unlink()
        self.shm_index.close()
        self.shm_index.unlink()

    @staticmethod
    def attach_existing(
        frame_shape: FrameShape, buffer_size=BufferSize, shm_name=None
    ) -> "SharedBuffer":
        return SharedBuffer(frame_shape, buffer_size=buffer_size, shm_name=shm_name)
