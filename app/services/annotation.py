"""Drawing helpers for overlaying predictions on frames."""

import cv2
import numpy as np


def annotate(image: np.ndarray, top5: list) -> np.ndarray:
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
