from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.streamer import VideoStreamer

router = APIRouter()

shared_buffer = None  # Shared buffer to be initialized globally


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    streamer = VideoStreamer(shared_buffer)
    await streamer.stream_frames(websocket)
