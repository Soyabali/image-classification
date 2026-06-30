"""Image classification inference helpers."""

import cv2
import numpy as np

from app.core.model import model


def preprocess(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, (model.imgsz, model.imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def infer(image: np.ndarray):
    """Run the model on a single BGR image.

    Returns: (top1_class, top1_confidence, top5_predictions)
    """
    raw = model.session.run(None, {model.input_name: preprocess(image)})[0][0]
    probs = softmax(raw)
    top5_idx = np.argsort(probs)[::-1][:5]
    top1_class = model.names[int(top5_idx[0])]
    top1_conf = round(float(probs[top5_idx[0]]), 4)
    top5 = [
        {"class": model.names[int(i)], "confidence": round(float(probs[i]), 4)}
        for i in top5_idx
    ]
    return top1_class, top1_conf, top5
