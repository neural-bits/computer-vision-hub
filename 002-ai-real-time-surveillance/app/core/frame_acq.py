import time

import av
import cv2
import numpy as np

from core.dtypes import FrameBuffer


class FrameAcquisition:
    def __init__(self, shared_buffer: FrameBuffer, video_path):
        self.shared_buffer = shared_buffer
        self.video_path = video_path
        print("Attached to shared memory", self.shared_buffer.shm.name)

    def acquire_frames(self):
        capture = av.open(self.video_path)
        for packet in capture.demux():
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    # Wait until there is space in the buffer
                    self.shared_buffer.space_available.acquire()

                    # Determine next index for the circular buffer
                    next_index = (
                        self.shared_buffer.latest_index.value + 1
                    ) % self.shared_buffer.buffer_size
                    image = frame.to_rgb().to_ndarray()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Write the frame to the shared memory buffer
                    np.copyto(self.shared_buffer.buffer[next_index], image)
                    self.shared_buffer.index_buffer[next_index] = next_index
                    self.shared_buffer.latest_index.value = next_index

                    # Signal that a new frame is available
                    print(f"Produced frame {next_index}")
                    # 1 frame is available now in the buffer, we release the sem so the consumer can consume it
                    self.shared_buffer.frame_available.release()

                    time.sleep(1 / 30)
