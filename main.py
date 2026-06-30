from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
import cv2
import numpy as np
import ast
import json
import tempfile
import os
import shutil
import uuid

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(
    title="Weather Image Classification API",
    description="Classify weather conditions in images and videos using YOLOv11",
    version="1.0.0"
)

app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

MODEL_PATH = "best.onnx"
_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name
_imgsz = _session.get_inputs()[0].shape[2]  # 224
_names: dict = ast.literal_eval(
    _session.get_modelmeta().custom_metadata_map.get("names", "{}")
)


def _preprocess(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, (_imgsz, _imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _infer(image: np.ndarray):
    raw = _session.run(None, {_input_name: _preprocess(image)})[0][0]
    probs = _softmax(raw)
    top5_idx = np.argsort(probs)[::-1][:5]
    top1_class = _names[int(top5_idx[0])]
    top1_conf = round(float(probs[top5_idx[0]]), 4)
    top5 = [
        {"class": _names[int(i)], "confidence": round(float(probs[i]), 4)}
        for i in top5_idx
    ]
    return top1_class, top1_conf, top5


def _annotate(image: np.ndarray, top5: list) -> np.ndarray:
    """Draw all top-5 predictions on the image with 20px padding on every side."""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, w / 1000)
    thickness = max(1, int(scale * 2))
    pad = 20
    line_gap = 8

    lines = [
        f"{'>' if i == 0 else ' '} {p['class']}  {p['confidence']:.1%}"
        for i, p in enumerate(top5)
    ]

    sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    max_tw = max(s[0][0] for s in sizes)
    line_h = max(s[0][1] for s in sizes)

    box_w = pad + max_tw + pad
    box_h = pad + len(lines) * line_h + (len(lines) - 1) * line_gap + pad

    # Semi-transparent dark background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, image, 0.30, 0, image)

    y = pad + line_h
    for i, line in enumerate(lines):
        color = (0, 255, 80) if i == 0 else (210, 210, 210)
        cv2.putText(image, line, (pad, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_h + line_gap

    return image


@app.get("/")
def root():
    return {"message": "Weather Image Classification API is running", "docs": "/docs"}


@app.post("/predict/image")
async def predict_image(request: Request, file: UploadFile = File(...)):
    """
    Upload an image and get weather classification results plus an annotated image URL.
    Returns the top predicted class, confidence, top-5 predictions, and a URL to the annotated image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image — ensure it is a valid image file")

    top1_class, top1_conf, top5 = _infer(image)
    annotated = _annotate(image.copy(), top5)

    filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(output_path, annotated)

    base_url = str(request.base_url).rstrip("/")
    image_url = f"{base_url}/results/{filename}"

    return JSONResponse({
        "filename": file.filename,
        "prediction": top1_class,
        "confidence": top1_conf,
        "top5_predictions": top5,
        "annotated_image_url": image_url
    })


@app.post("/predict/image/annotated")
async def predict_image_annotated(file: UploadFile = File(...)):
    """
    Upload an image and get back the annotated image file directly (with all top-5 labels drawn).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image — ensure it is a valid image file")

    top1_class, top1_conf, top5 = _infer(image)
    annotated = _annotate(image.copy(), top5)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, annotated)
    tmp.close()

    return FileResponse(tmp.name, media_type="image/jpeg", filename="result.jpg")


@app.post("/predict/video")
async def predict_video(request: Request, file: UploadFile = File(...)):
    """
    Upload a video (.mp4 or .avi) and get back:
    - A URL to the annotated video with top-5 classification labels drawn on each frame
    - Dominant predicted class across all frames
    - Per-class frame counts
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    suffix = os.path.splitext(file.filename)[-1] or ".avi"
    tmp_input = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)

    out_filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(RESULTS_DIR, out_filename)

    try:
        shutil.copyfileobj(file.file, tmp_input)
        tmp_input.flush()
        tmp_input.close()

        cap = cv2.VideoCapture(tmp_input.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        class_counts: dict[str, int] = {}
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            top1_class, top1_conf, top5 = _infer(frame)
            class_counts[top1_class] = class_counts.get(top1_class, 0) + 1
            frame_count += 1

            writer.write(_annotate(frame.copy(), top5))

        cap.release()
        writer.release()

        if frame_count == 0:
            raise HTTPException(status_code=400, detail="Video has no readable frames")

        dominant_class = max(class_counts, key=class_counts.get)
        base_url = str(request.base_url).rstrip("/")
        video_url = f"{base_url}/results/{out_filename}"

        return JSONResponse({
            "filename": file.filename,
            "total_frames": frame_count,
            "dominant_class": dominant_class,
            "class_distribution": class_counts,
            "annotated_video_url": video_url
        })

    finally:
        if os.path.exists(tmp_input.name):
            os.unlink(tmp_input.name)


@app.post("/predict/video/stream")
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

    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    tmp_input = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    shutil.copyfileobj(file.file, tmp_input)
    tmp_input.flush()
    tmp_input.close()
    tmp_path = tmp_input.name
    original_filename = file.filename

    out_filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(RESULTS_DIR, out_filename)
    base_url = str(request.base_url).rstrip("/")
    video_url = f"{base_url}/results/{out_filename}"

    async def generate():
        writer = None
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                yield f"data: {json.dumps({'event': 'error', 'message': 'Could not open video file'})}\n\n"
                return

            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            yield f"data: {json.dumps({'event': 'start', 'filename': original_filename, 'total_frames': total})}\n\n"

            class_counts: dict = {}
            frame_num = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                top1_class, top1_conf, top5 = _infer(frame)
                class_counts[top1_class] = class_counts.get(top1_class, 0) + 1
                frame_num += 1

                writer.write(_annotate(frame.copy(), top5))

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

    return StreamingResponse(generate(), media_type="text/event-stream")
