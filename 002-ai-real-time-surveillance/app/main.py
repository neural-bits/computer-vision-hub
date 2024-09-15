import multiprocessing as mp
from multiprocessing import Process

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import video_stream

from core.buffer import SharedBuffer
from core.frame_acq import FrameAcquisition
from core.inference import InferenceHandler
from core.streamer import VideoStreamer

# Global variables for processes and buffer
shared_buffer = None
acquisition_process = None
inference_process = None

cv2.setNumThreads(1)


def main():
    global shared_buffer, acquisition_process, inference_process
    mp.set_start_method("fork", force=True)

    # Define shared buffer
    frame_shape = (720, 1280, 3)
    buffer_size = 10
    shared_buffer = SharedBuffer(frame_shape, buffer_size)

    # Start acquisition and inference processes
    acquisition = FrameAcquisition(
        shared_buffer,
        "/Users/razvantalexandru/Desktop/NeuralBits/Projects/NeuralBitsNewsletter/hub/computer-vision-hub/002-ai-real-time-surveillance/core/test_video.mp4",
    )
    streamer = VideoStreamer(shared_buffer)
    infer = InferenceHandler(shared_buffer)

    acquisition_process = Process(target=acquisition.acquire_frames)
    streamer_process = Process(target=streamer.stream_frames)
    inference_process1 = Process(target=infer.execute)
    inference_process2 = Process(target=infer.execute)
    inference_process3 = Process(target=infer.execute)
    inference_process4 = Process(target=infer.execute)

    acquisition_process.start()
    streamer_process.start()
    inference_process1.start()
    inference_process2.start()
    inference_process3.start()
    inference_process4.start()

    acquisition_process.join()
    streamer_process.join()
    inference_process1.join()
    inference_process2.join()
    inference_process3.join()
    inference_process4.join()

    # When the application shuts down, clean up resources
    if acquisition_process and acquisition_process.is_alive():
        acquisition_process.terminate()
        acquisition_process.join()

    # if inference_process and inference_process.is_alive():
    # inference_process.terminate()
    # inference_process.join()

    shared_buffer.cleanup()


if __name__ == "__main__":
    main()
