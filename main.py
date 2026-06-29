from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
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

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


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

    results = model(image)[0] # to get the first result from the batch
    probs = results.probs

    top1_class = results.names[int(probs.top1)]
    top1_conf = round(float(probs.top1conf), 4)

    top5 = [
        {"class": results.names[int(idx)], "confidence": round(float(conf), 4)}
        for idx, conf in zip(probs.top5, probs.top5conf)
    ]

    annotated = results.plot()
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
    Upload an image and get back the annotated image file directly (with classification label drawn).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image — ensure it is a valid image file")

    results = model(image)[0]
    annotated = results.plot()

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, annotated)
    tmp.close()

    return FileResponse(tmp.name, media_type="image/jpeg", filename="result.jpg")


@app.post("/predict/video")
async def predict_video(request: Request, file: UploadFile = File(...)):
    """
    Upload a video (.mp4 or .avi) and get back:
    - A URL to the annotated video with classification labels drawn on each frame
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

            results = model(frame)[0]
            top1_class = results.names[int(results.probs.top1)]
            class_counts[top1_class] = class_counts.get(top1_class, 0) + 1
            frame_count += 1

            writer.write(results.plot())

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
