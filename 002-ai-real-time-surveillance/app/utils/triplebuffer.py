from abc import ABC, abstractmethod

from torch import multiprocessing as mp


class TripleBufferABC(mp.Process, ABC):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def setup_buffers(self):
        pass
