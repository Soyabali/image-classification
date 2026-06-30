"""Application entry point: builds the FastAPI app and wires everything together.

Run with:  uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import health, image, video

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)

# Serve annotated images/videos from the results directory.
app.mount("/results", StaticFiles(directory=settings.RESULTS_DIR), name="results")

# Register routes.
app.include_router(health.router)
app.include_router(image.router)
app.include_router(video.router)
