from core.buffer import SharedBuffer


class InferenceHandler:
    def __init__(self, shared_buffer: SharedBuffer = None):
        self.shm_buffer = shared_buffer

    def process(self):
        while True:
            # Acq shared mem semaphore
            self.shm_buffer.frame_available.acquire()

            current_index = self.shm_buffer.latest_index.value
            frame = self.shm_buffer.buffer[current_index]
            print(f"Inference on buffer active, im shape {frame.shape}")

            self.shm_buffer.index_buffer[current_index] = -1
            self.shm_buffer.space_available.release()
