import anyio.to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from inference.image_index import image_index

from .predictor import get_predictor
from .schema import (
    ClassifyRequest,
    ClassifyResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
)
from .utils import run_detect, stream_file_to_upload, stream_url_to_upload

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        stage1_model_loaded=predictor.stage1_loaded if predictor else False,
        stage2_model_loaded=predictor.stage2_loaded if predictor else False,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect_url(request: DetectRequest) -> DetectResponse:
    """Detect screen photo from image URL."""

    try:
        entry = await stream_url_to_upload(request.url)
        return await anyio.to_thread.run_sync(run_detect, entry.path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc!r}",
        ) from exc


@router.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(file: UploadFile = File()) -> DetectResponse:
    """Detect screen photo from uploaded file."""
    try:
        entry = await stream_file_to_upload(file)
        return await anyio.to_thread.run_sync(run_detect, entry.path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc!r}",
        ) from exc


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(request: ClassifyRequest) -> ClassifyResponse:
    class_name = "screen_photo" if request.is_screen else "normal_photo"

    try:
        await image_index.classify(request.image_id, class_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found: {request.image_id}",
        ) from exc

    return ClassifyResponse(
        image_id=request.image_id,
        is_screen=request.is_screen,
        class_name=class_name,
    )
