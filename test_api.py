"""
Quick test script — run this to check if image and video endpoints work.

Usage:
    python test_api.py                          # tests with a generated image
    python test_api.py --image path/to/img.jpg  # test with your own image
    python test_api.py --video path/to/vid.mp4  # test with your own video
"""

import requests
import argparse
import numpy as np
import cv2
import tempfile
import os

BASE_URL = "http://localhost:8000"


def test_root():
    print("\n--- Testing root endpoint ---")
    r = requests.get(f"{BASE_URL}/")
    print("Status:", r.status_code)
    print("Response:", r.json())


def test_image(image_path: str):
    print(f"\n--- Testing /predict/image with: {image_path} ---")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/image",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")}
        )
    print("Status:", r.status_code)
    print("Response:", r.json())


def test_image_annotated(image_path: str):
    print(f"\n--- Testing /predict/image/annotated with: {image_path} ---")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/image/annotated",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")}
        )
    print("Status:", r.status_code)
    if r.status_code == 200:
        out = "annotated_result.jpg"
        with open(out, "wb") as f:
            f.write(r.content)
        print(f"Annotated image saved as: {out}")
    else:
        print("Error:", r.text)


def test_video(video_path: str):
    print(f"\n--- Testing /predict/video with: {video_path} ---")
    with open(video_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/video",
            files={"file": (os.path.basename(video_path), f, "video/mp4")}
        )
    print("Status:", r.status_code)
    if r.status_code == 200:
        out = "annotated_result.mp4"
        with open(out, "wb") as f:
            f.write(r.content)
        print(f"Annotated video saved as: {out}")
    else:
        print("Error:", r.text)


def create_dummy_image():
    """Creates a simple test image if you don't have one."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    cv2.putText(img, "Test Image", (220, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(path, img)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to a test image")
    parser.add_argument("--video", help="Path to a test video")
    args = parser.parse_args()

    test_root()

    image_path = args.image
    if not image_path:
        print("\nNo image provided — generating a dummy test image...")
        image_path = create_dummy_image()

    test_image(image_path)
    test_image_annotated(image_path)

    if args.video:
        test_video(args.video)
    else:
        print("\nSkipping video test — pass --video path/to/video.mp4 to test it")
