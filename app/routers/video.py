"""Video classification endpoints (full annotated video + live SSE stream)."""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.services.video_service import (
    generate_stream,
    new_output_path,
    process_video,
    save_upload_to_tempfile,
)

router = APIRouter(prefix="/predict", tags=["video"])


@router.post("/video")
async def predict_video(request: Request, file: UploadFile = File(...)):
    """
    Upload a video (.mp4 or .avi) and get back:
    - A URL to the annotated video with top-5 classification labels drawn on each frame
    - Dominant predicted class across all frames
    - Per-class frame counts
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    tmp_path = save_upload_to_tempfile(file, default_suffix=".avi")
    out_filename, output_path = new_output_path()

    result = process_video(tmp_path, output_path, file.filename)

    base_url = str(request.base_url).rstrip("/")
    result["annotated_video_url"] = f"{base_url}/results/{out_filename}"

    return JSONResponse(result)


@router.post("/video/stream")
async def predict_video_stream(request: Request, file: UploadFile = File(...)):
    """
    Upload a video (.mp4 or .avi) and receive:
    - Live frame-by-frame predictions as Server-Sent Events (SSE)
    - A playable annotated video URL (with top-5 labels on every frame) in the final `done` event

    SSE events:
    - `start` — filename, total_frames
    - `frame` — frame number, top1 class & confidence, full top5 list
    - `done`  — dominant_class, class_distribution, total_frames, annotated_video_url
    - `error` — message on failure

    Flutter: listen to the stream, parse each `data: {...}` line, and play the URL from `done`.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    tmp_path = save_upload_to_tempfile(file, default_suffix=".mp4")
    out_filename, output_path = new_output_path()

    base_url = str(request.base_url).rstrip("/")
    video_url = f"{base_url}/results/{out_filename}"

    return StreamingResponse(
        generate_stream(tmp_path, output_path, file.filename, video_url),
        media_type="text/event-stream",
    )
