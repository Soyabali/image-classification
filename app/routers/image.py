"""Image classification endpoints."""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.services.image_service import (
    classify_and_save,
    classify_to_tempfile,
    decode_image,
)

router = APIRouter(prefix="/predict", tags=["image"])


@router.post("/image")
async def predict_image(request: Request, file: UploadFile = File(...)):
    """
    Upload an image and get weather classification results plus an annotated image URL.
    Returns the top predicted class, confidence, top-5 predictions, and a URL to the annotated image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = decode_image(contents)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image — ensure it is a valid image file",
        )

    top1_class, top1_conf, top5, filename = classify_and_save(image)

    base_url = str(request.base_url).rstrip("/")
    image_url = f"{base_url}/results/{filename}"

    return JSONResponse(
        {
            "filename": file.filename,
            "prediction": top1_class,
            "confidence": top1_conf,
            "top5_predictions": top5,
            "annotated_image_url": image_url,
        }
    )


@router.post("/image/annotated")
async def predict_image_annotated(file: UploadFile = File(...)):
    """
    Upload an image and get back the annotated image file directly (with all top-5 labels drawn).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = decode_image(contents)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image — ensure it is a valid image file",
        )

    tmp_path = classify_to_tempfile(image)
    return FileResponse(tmp_path, media_type="image/jpeg", filename="result.jpg")
