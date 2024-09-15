import time

import cv2

from core.dtypes import FrameBuffer


class VideoStreamer:
    def __init__(self, shared_buffer: FrameBuffer):
        self.shared_buffer = shared_buffer
        self.last_time = time.time()
        self.frame_count = 0

    def calculate_fps(self):
        """Calculate the dynamic FPS."""
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        if elapsed_time > 0:
            fps = 1 / elapsed_time
        else:
            fps = 0
        return fps

    def stream_frames(self):
        try:
            while True:
                # Wait for the frame to be available from acquisition
                self.shared_buffer.frame_available.acquire()

                # Get the latest frame
                current_index = self.shared_buffer.latest_index.value
                frame = self.shared_buffer.buffer[current_index]
                fps = self.calculate_fps()

                # Add FPS to the frame
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                print("Added FPS to frame")
                # Mark frame as consumed and release space
                self.shm_buffer.index_buffer[current_index] = -1
                self.shm_buffer.space_available.release()

        finally:
            cv2.destroyAllWindows()
