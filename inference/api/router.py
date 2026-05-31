import io
import zipfile
from datetime import UTC, datetime
from typing import Annotated

import anyio.to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from ..config import UPLOAD_DIR
from ..image_index import image_index
from .predictor import get_predictor
from .schema import (
    ClassifyRequest,
    ClassifyResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    PackageRequest,
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
        is_screen = await anyio.to_thread.run_sync(run_detect, entry.path)
        await image_index.classify(entry.file_hash, is_screen)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc!r}",
        ) from exc
    else:
        return DetectResponse(image_id=entry.file_hash, is_screen=is_screen)


@router.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(
    file: Annotated[UploadFile, File()],
) -> DetectResponse:
    """Detect screen photo from uploaded file."""
    try:
        entry = await stream_file_to_upload(file)
        is_screen = await anyio.to_thread.run_sync(run_detect, entry.path)
        await image_index.classify(entry.file_hash, is_screen)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc!r}",
        ) from exc
    else:
        return DetectResponse(image_id=entry.file_hash, is_screen=is_screen)


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(request: ClassifyRequest) -> ClassifyResponse:
    try:
        entry = await image_index.classify(request.image_id, request.is_screen)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found: {request.image_id}",
        ) from exc

    return ClassifyResponse(
        image_id=request.image_id,
        is_screen=request.is_screen,
        class_name=entry.class_name or "unclassified",
    )


@router.post("/package")
async def package_images(request: PackageRequest) -> StreamingResponse:
    """Package images uploaded after the given timestamp into a zip file.

    Returns a zip file containing:
    - /screen_photo/*: screen capture images
    - /normal_photo/*: non-screen images
    """
    # Filter images after timestamp
    after_time = (
        after_time.astimezone(UTC)
        if (after_time := request.after_timestamp).tzinfo
        else after_time.replace(tzinfo=UTC)
    )

    async with image_index.load_index() as index:
        matching_entries = [
            entry
            for entry in index.values()
            if entry.created_at > after_time and entry.path.exists()
        ]

        if not matching_entries:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found after the specified timestamp",
            )

        # Create zip in memory with classified images sorted into folders
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in matching_entries:
                zf.write(entry.path, entry.path.relative_to(UPLOAD_DIR))
        zip_buffer.seek(0)

    # Generate filename with timestamp
    zip_filename = f"images_{datetime.now(UTC):%Y%m%d_%H%M%S}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
    )
