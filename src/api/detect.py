"""FastAPI 路由 - 图像检测 API"""

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
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


def _suffix_from_content_type(content_type: str) -> str:
    for mime, ext in CONTENT_TYPE_SUFFIX_MAP.items():
        if mime in content_type:
            return ext
    return ".jpg"


def _remove_temp_file(path: Path) -> None:
    path.unlink(missing_ok=True)


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
            resp = await client.get(url)
            resp.raise_for_status()
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

    content_type = resp.headers.get("content-type", "")
    logger.info("Content-Type: %s, Size: %d bytes", content_type, len(resp.content))

    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Not an image (got {content_type})",
        )

    suffix = _suffix_from_content_type(content_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(resp.content)
        tmp_path = Path(tmp.name)

    try:
        result = await asyncio.to_thread(_detector.detect, tmp_path)
        is_screen = result["result"] in ("screenshot", "screen_photo")
        logger.info("Result: %s (is_screen=%s)", result["result"], is_screen)
        return DetectResponse(is_screen=is_screen)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Unable to process the image file",
        ) from None
    finally:
        _remove_temp_file(tmp_path)
