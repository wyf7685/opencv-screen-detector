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
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(str(req.url))
            resp.raise_for_status()
        except httpx.HTTPError as err:
            raise HTTPException(
                status_code=502,
                detail="Failed to download image from URL",
            ) from err

    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail="URL does not point to an image",
        )

    suffix = _suffix_from_content_type(content_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(resp.content)
        tmp_path = Path(tmp.name)

    try:
        result = await asyncio.to_thread(_detector.detect, tmp_path)
        is_screen = result["result"] in ("screenshot", "screen_photo")
        return DetectResponse(is_screen=is_screen)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Unable to process the image file",
        ) from None
    finally:
        _remove_temp_file(tmp_path)
