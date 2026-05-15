"""FastAPI 路由 - 图像检测 API"""

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterable
from pathlib import Path

import fleep
import httpx
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from src.detector import ScreenDetector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["detect"])

_detector = ScreenDetector()

CONTENT_TYPE_SUFFIX_MAP: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}
FLEEP_HEADER_SIZE = 128
STREAM_CHUNK_SIZE = 64 * 1024  # 64 KB

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


class DetectRequest(BaseModel):
    url: HttpUrl


class DetectResponse(BaseModel):
    is_screen: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def _get_image_ext(header: bytes) -> str | None:
    if len(header) < FLEEP_HEADER_SIZE:
        return None
    info = fleep.get(header[:FLEEP_HEADER_SIZE])
    return CONTENT_TYPE_SUFFIX_MAP.get(info.mime[0]) if info.mime else None


async def _stream_to_temp(
    stream: AsyncIterable[bytes],
    first_chunk: bytes,
    suffix: str,
) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        await asyncio.to_thread(tmp.write, first_chunk)
        async for chunk in stream:
            await asyncio.to_thread(tmp.write, chunk)
        await asyncio.to_thread(tmp.flush)
        return Path(tmp.name)


async def _stream_file(file: UploadFile) -> AsyncIterable[bytes]:
    while chunk := await file.read(STREAM_CHUNK_SIZE):
        yield chunk


async def _run_detector(path: Path) -> DetectResponse:
    try:
        result = await asyncio.to_thread(_detector.detect, path)
        is_screen = result["result"] in ("screenshot", "screen_photo")
        logger.info("Result: %s (is_screen=%s)", result["result"], is_screen)
        return DetectResponse(is_screen=is_screen)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Unable to process the image file",
        ) from None
    finally:
        path.unlink(missing_ok=True)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_detector._ml_model.is_loaded,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect_image(req: DetectRequest) -> DetectResponse:
    url = str(req.url)
    logger.info("Downloading image from: %s", url)

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers=HEADERS,
    ) as client:
        try:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()

                aiter = resp.aiter_bytes(STREAM_CHUNK_SIZE)
                first_chunk = b""
                async for chunk in aiter:
                    first_chunk += chunk
                    if len(first_chunk) >= FLEEP_HEADER_SIZE:
                        break

                if not (suffix := _get_image_ext(first_chunk)):
                    raise HTTPException(status_code=422, detail="Not an image")

                tmp_path = await _stream_to_temp(aiter, first_chunk, suffix)
                logger.info("Saved upload to temp file: %s", tmp_path)

        except httpx.HTTPStatusError as err:
            logger.exception("HTTP error %d", err.response.status_code)
            raise HTTPException(
                status_code=502,
                detail=f"HTTP {err.response.status_code}",
            ) from err
        except httpx.HTTPError as err:
            logger.exception("Download error")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download: {err}",
            ) from err

    return await _run_detector(tmp_path)


@router.post("/detect/upload", response_model=DetectResponse)
async def detect_upload(file: UploadFile) -> DetectResponse:
    logger.info("Receiving uploaded file: %s", file.filename)

    first_chunk = await file.read(FLEEP_HEADER_SIZE)
    if not first_chunk:
        raise HTTPException(status_code=400, detail="Empty file")

    if not (suffix := _get_image_ext(first_chunk)):
        raise HTTPException(
            status_code=422,
            detail="Not an image file",
        )

    tmp_path = await _stream_to_temp(_stream_file(file), first_chunk, suffix)
    logger.info("Saved upload to temp file: %s", tmp_path)

    return await _run_detector(tmp_path)
