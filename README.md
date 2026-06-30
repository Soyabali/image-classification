# Weather Image Classification API

A FastAPI service that classifies weather conditions in **images** and **videos**
using a YOLOv11 classification model (ONNX runtime, CPU).

## Project structure

```
imageclassification_Api/
├── app/                      # Application package
│   ├── main.py               # FastAPI app: builds app, mounts static, wires routers
│   ├── config.py             # Central settings (paths, model location, API metadata)
│   ├── core/
│   │   └── model.py          # Loads the ONNX model once (shared singleton)
│   ├── services/             # Business logic — no HTTP code here
│   │   ├── inference.py      # preprocess · softmax · infer
│   │   ├── annotation.py     # draw top-5 overlay on a frame
│   │   ├── image_service.py  # image pipeline (decode → infer → annotate → save)
│   │   └── video_service.py  # video pipeline + live SSE streaming
│   └── routers/              # HTTP endpoints only
│       ├── health.py         # GET  /
│       ├── image.py          # POST /predict/image, /predict/image/annotated
│       └── video.py          # POST /predict/video, /predict/video/stream
├── models/                   # Model weights
│   ├── best.onnx             # served by the API
│   └── best.pt               # original PyTorch weights
├── results/                  # Runtime output (annotated images/videos) — git-ignored
├── tests/
│   └── test_api.py           # Manual endpoint test script
├── requirements.txt
├── render.yaml               # Render.com deploy config
└── README.md
```

### How the layers fit together

```
routers  →  services  →  core/model
(HTTP)      (logic)       (the loaded model)
```

- **routers** only deal with HTTP: validate the upload, call a service, shape the
  response. Adding a new endpoint means adding a router function.
- **services** hold the actual work (inference, annotation, video handling) and
  know nothing about FastAPI requests.
- **core/model.py** loads the model a single time and is reused everywhere.
- **config.py** is the one place paths and settings live.

## Run locally

```bash
# from the project root, with your virtualenv active
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for interactive API docs.

## Endpoints

| Method | Path                      | Description                                            |
|--------|---------------------------|--------------------------------------------------------|
| GET    | `/`                       | Health check                                           |
| POST   | `/predict/image`          | Classify an image → JSON + annotated image URL         |
| POST   | `/predict/image/annotated`| Classify an image → annotated image file directly      |
| POST   | `/predict/video`          | Classify a video → JSON + annotated video URL          |
| POST   | `/predict/video/stream`   | Classify a video → live per-frame predictions via SSE  |

Annotated results are written to `results/` and served from `/results/<file>`.

## Test

```bash
python tests/test_api.py                          # uses a generated dummy image
python tests/test_api.py --image path/to/img.jpg  # your own image
python tests/test_api.py --video path/to/vid.mp4  # your own video
```
