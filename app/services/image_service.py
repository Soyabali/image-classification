"""Image processing pipeline: decode -> infer -> annotate -> save."""

import os
import tempfile
import uuid

import cv2
import numpy as np

from app.config import settings
from app.services.annotation import annotate
from app.services.inference import infer


def decode_image(contents: bytes) -> np.ndarray | None:
    """Decode raw bytes into a BGR image, or None if it is not a valid image."""
    np_array = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def classify_and_save(image: np.ndarray):
    """Classify the image, annotate a copy, save it, and return the result.

    Returns: (top1_class, top1_confidence, top5_predictions, filename)
    """
    top1_class, top1_conf, top5 = infer(image)
    annotated = annotate(image.copy(), top5)

    filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(settings.RESULTS_DIR, filename)
    cv2.imwrite(output_path, annotated)

    return top1_class, top1_conf, top5, filename


def classify_to_tempfile(image: np.ndarray) -> str:
    """Classify and annotate the image, writing it to a temp file.

    Returns: the path to the annotated JPEG.
    """
    _, _, top5 = infer(image)
    annotated = annotate(image.copy(), top5)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, annotated)
    tmp.close()
    return tmp.name
