import random

import cv2

from core.buffer import SharedBuffer


class InferenceHandler:
    def __init__(
        self,
        shared_buffer: SharedBuffer = None,
    ):
        self.shm_buffer = shared_buffer
        self.options = ["alan", "dala", "coco", "jumbo"]

    def execute(self):
        try:
            while True:
                # Acquire shared memory semaphore
                self.shm_buffer.frame_available.acquire()

                # Get the latest frame
                current_index = self.shm_buffer.latest_index.value
                frame = self.shm_buffer.buffer[current_index]

                # Get the center of the frame
                height, width, _ = frame.shape
                center_x, center_y = width // 2, height // 2
                text = random.choice(self.options)
                # Add text to the center of the frame
                cv2.putText(
                    frame,
                    text,
                    (center_x - 100, center_y),  # Adjust for text alignment
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Inference", frame)
                cv2.waitKey(1)

                # Mark frame as consumed and release space
                self.shm_buffer.index_buffer[current_index] = -1
                self.shm_buffer.space_available.release()

        finally:
            # Release resources and close OpenCV windows
            cv2.destroyAllWindows()
