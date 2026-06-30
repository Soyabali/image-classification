"""Health / root endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"message": "Weather Image Classification API is running", "docs": "/docs"}
