import multiprocessing as mp
import multiprocessing.shared_memory as shm
from typing import Tuple

import numpy as np
from numpy import typing as npt

FrameShape = Tuple[int, int, int]
BufferSize = int

# SharedBuffer-related types
SemaphoreType = mp.Semaphore
SharedMemoryType = shm.SharedMemory
SharedMemoryValue = mp.Value
FrameBuffer = npt.NDArray[np.uint8]
IndexBuffer = npt.NDArray[np.int32]
