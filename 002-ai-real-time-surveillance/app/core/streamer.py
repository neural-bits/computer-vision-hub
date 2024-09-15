import cv2

from core.dtypes import FrameBuffer


class VideoStreamer:
    def __init__(self, shared_buffer: FrameBuffer):
        self.shared_buffer = shared_buffer

    def stream_frames(self):
        try:
            while True:
                self.shared_buffer.frame_available.acquire()

                # Get the latest frame
                current_index = self.shared_buffer.latest_index.value
                frame = self.shared_buffer.buffer[current_index]

                # Display the frame using OpenCV
                cv2.imshow("Demo", frame)

                # Wait for a short time (1 ms) and break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting the video stream...")
                    break

                # Mark frame as consumed
                self.shared_buffer.index_buffer[current_index] = -1
                self.shared_buffer.space_available.release()
        finally:
            # Release resources and close any OpenCV windows
            cv2.destroyAllWindows()
