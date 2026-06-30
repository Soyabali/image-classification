"""Video processing pipeline: per-frame inference, annotated output and SSE stream."""

import json
import os
import shutil
import tempfile
import uuid

import cv2
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.services.annotation import annotate
from app.services.inference import infer


def save_upload_to_tempfile(file: UploadFile, default_suffix: str) -> str:
    """Persist an uploaded video to a temp file and return its path."""
    suffix = os.path.splitext(file.filename)[-1] or default_suffix
    tmp_input = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    shutil.copyfileobj(file.file, tmp_input)
    tmp_input.flush()
    tmp_input.close()
    return tmp_input.name


def new_output_path() -> tuple[str, str]:
    """Return (filename, absolute_path) for a new annotated video in RESULTS_DIR."""
    out_filename = f"{uuid.uuid4()}.mp4"
    return out_filename, os.path.join(settings.RESULTS_DIR, out_filename)


def process_video(tmp_path: str, output_path: str, original_filename: str) -> dict:
    """Annotate an entire video and return summary data.

    Raises HTTPException on unreadable input.
    """
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        class_counts: dict[str, int] = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            top1_class, top1_conf, top5 = infer(frame)
            class_counts[top1_class] = class_counts.get(top1_class, 0) + 1
            frame_count += 1

            writer.write(annotate(frame.copy(), top5))

        cap.release()
        writer.release()

        if frame_count == 0:
            raise HTTPException(status_code=400, detail="Video has no readable frames")

        dominant_class = max(class_counts, key=class_counts.get)

        return {
            "filename": original_filename,
            "total_frames": frame_count,
            "dominant_class": dominant_class,
            "class_distribution": class_counts,
        }

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def generate_stream(
    tmp_path: str, output_path: str, original_filename: str, video_url: str
):
    """Yield Server-Sent Events while annotating the video frame by frame."""
    writer = None
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            yield f"data: {json.dumps({'event': 'error', 'message': 'Could not open video file'})}\n\n"
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        yield f"data: {json.dumps({'event': 'start', 'filename': original_filename, 'total_frames': total})}\n\n"

        class_counts: dict = {}
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            top1_class, top1_conf, top5 = infer(frame)
            class_counts[top1_class] = class_counts.get(top1_class, 0) + 1
            frame_num += 1

            writer.write(annotate(frame.copy(), top5))

            yield f"data: {json.dumps({'event': 'frame', 'frame': frame_num, 'class': top1_class, 'confidence': top1_conf, 'top5': top5})}\n\n"

        cap.release()
        writer.release()
        writer = None

        if frame_num == 0:
            yield f"data: {json.dumps({'event': 'error', 'message': 'Video has no readable frames'})}\n\n"
            return

        dominant_class = max(class_counts, key=class_counts.get)
        yield f"data: {json.dumps({'event': 'done', 'total_frames': frame_num, 'dominant_class': dominant_class, 'class_distribution': class_counts, 'annotated_video_url': video_url})}\n\n"

    finally:
        if writer is not None:
            writer.release()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
