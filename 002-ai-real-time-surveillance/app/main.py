import multiprocessing as mp
import signal
from contextlib import asynccontextmanager
from multiprocessing import Process

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize everything here during startup
    global shared_buffer, acquisition_process, inference_process

    # Define shared buffer
    frame_shape = (720, 1280, 3)
    buffer_size = 10
    shared_buffer = SharedBuffer(frame_shape, buffer_size)

    # Pass the shared buffer to the WebSocket route
    video_stream.shared_buffer = shared_buffer

    # Start acquisition and inference processes
    acquisition = FrameAcquisition(
        shared_buffer,
        "/Users/razvantalexandru/Desktop/NeuralBits/Projects/NeuralBitsNewsletter/hub/computer-vision-hub/002-ai-real-time-surveillance/core/test_video.mp4",
    )
    inference = InferenceHandler(shared_buffer=shared_buffer)

    acquisition_process = Process(target=acquisition.acquire_frames)
    acquisition_process.start()

    # inference_process = Process(target=inference.process)
    # inference_process.start()
    # Yield control back to the application while these processes are running
    yield

    # When the application shuts down, clean up resources
    if acquisition_process and acquisition_process.is_alive():
        acquisition_process.terminate()
        acquisition_process.join()

    # if inference_process and inference_process.is_alive():
    # inference_process.terminate()
    # inference_process.join()

    shared_buffer.cleanup()


def signal_handler(sig, frame):
    global acquisition_process, inference_process, shared_buffer
    if acquisition_process:
        acquisition_process.terminate()
        acquisition_process.join()
    shared_buffer.cleanup()
    exit(0)


def main_ws():
    import multiprocessing as mp

    import uvicorn

    # Initialize FastAPI application
    app = FastAPI(
        title="video-streaming-api",
        description="Video streaming API",
        debug=True,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the routers
    app.include_router(video_stream.router)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)

    mp.set_start_method("fork", force=True)

    uvicorn.run("main:app", host="127.0.0.1", reload=True, port=8000)


def main():
    mp.set_start_method("fork", force=True)
    global shared_buffer, acquisition_process, inference_process

    # Define shared buffer
    frame_shape = (720, 1280, 3)
    buffer_size = 10
    shared_buffer = SharedBuffer(frame_shape, buffer_size)

    # Pass the shared buffer to the WebSocket route
    video_stream.shared_buffer = shared_buffer

    # Start acquisition and inference processes
    acquisition = FrameAcquisition(
        shared_buffer,
        "/Users/razvantalexandru/Desktop/NeuralBits/Projects/NeuralBitsNewsletter/hub/computer-vision-hub/002-ai-real-time-surveillance/core/test_video.mp4",
    )

    acquisition_process = Process(target=acquisition.acquire_frames)
    acquisition_process.start()
    acquisition_process.join()

    streamer = VideoStreamer(shared_buffer)
    streamer_process = Process(target=streamer.stream_frames)
    streamer_process.start()
    streamer_process.join()

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
