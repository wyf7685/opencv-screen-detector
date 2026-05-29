import io
import zipfile
from datetime import UTC, datetime

import anyio.to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from ..image_index import INDEX_FILE, ImageEntry, image_index
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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc!r}",
        ) from exc
    else:
        await image_index.classify(entry.file_hash, is_screen)
        return DetectResponse(image_id=entry.file_hash, is_screen=is_screen)


@router.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(file: UploadFile = File()) -> DetectResponse:
    """Detect screen photo from uploaded file."""
    try:
        entry = await stream_file_to_upload(file)
        is_screen = await anyio.to_thread.run_sync(run_detect, entry.path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc!r}",
        ) from exc
    else:
        await image_index.classify(entry.file_hash, is_screen)
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

    Returns a zip file containing all images created after the specified timestamp.
    """
    from pydantic import TypeAdapter

    # Load index
    ta = TypeAdapter(dict[str, ImageEntry])
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No image index found",
        )

    index = ta.validate_json(INDEX_FILE.read_bytes())

    # Filter images after timestamp
    after_time = request.after_timestamp
    if after_time.tzinfo is None:
        after_time = after_time.replace(tzinfo=UTC)

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

    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in matching_entries:
            zf.write(entry.path, entry.file_name)

    zip_buffer.seek(0)

    # Generate filename with timestamp
    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_filename = f"images_{timestamp_str}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
    )
