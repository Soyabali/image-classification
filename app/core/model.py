"""ONNX model wrapper.

The model is loaded exactly once when this module is first imported and reused
for every request, which keeps inference fast and memory usage low.
"""

import ast

import onnxruntime as ort

from app.config import settings


class ClassificationModel:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = self.session.get_inputs()[0].shape[2]  # 224
        self.names: dict = ast.literal_eval(
            self.session.get_modelmeta().custom_metadata_map.get("names", "{}")
        )


# Single shared instance used across the whole app.
model = ClassificationModel(settings.MODEL_PATH)
