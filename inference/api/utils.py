import hashlib
import tempfile
from collections.abc import AsyncIterable
from pathlib import Path

import anyio
import fleep
import httpx
from fastapi import HTTPException, UploadFile, status

from ..image_index import ImageEntry, image_index
from .predictor import get_predictor


async def _stream_to_temp(
    stream: AsyncIterable[bytes],
    first_chunk: bytes,
    suffix: str,
) -> tuple[str, Path]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)

    h = hashlib.sha256(first_chunk)
    async with await anyio.Path(tmp_path).open("wb") as file:
        await file.write(first_chunk)
        async for chunk in stream:
            h.update(chunk)
            await file.write(chunk)
        await file.flush()
    file_hash = h.hexdigest()

    return file_hash, tmp_path


CONTENT_TYPE_SUFFIX_MAP: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


def _get_image_ext(header: bytes) -> str | None:
    if len(header) < FLEEP_HEADER_SIZE:
        return None
    info = fleep.get(header[:FLEEP_HEADER_SIZE])
    return CONTENT_TYPE_SUFFIX_MAP.get(info.mime[0]) if info.mime else None


REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}
FLEEP_HEADER_SIZE = 128
STREAM_CHUNK_SIZE = 64 * 1024  # 64 KB


async def stream_url_to_upload(url: str) -> ImageEntry:
    try:
        async with (
            httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers=REQUEST_HEADERS,
            ) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            chunk_iter = resp.aiter_bytes(STREAM_CHUNK_SIZE)
            first_chunk = b""
            async for chunk in chunk_iter:
                first_chunk += chunk
                if len(first_chunk) >= FLEEP_HEADER_SIZE:
                    break
            if not (suffix := _get_image_ext(first_chunk)):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Not an image",
                )
            file_hash, tmp_path = await _stream_to_temp(
                chunk_iter, first_chunk, suffix
            )
            return await image_index.add(file_hash, tmp_path)
    except httpx.HTTPStatusError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HTTP {err.response.status_code}",
        ) from err
    except httpx.HTTPError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download: {err}",
        ) from err


async def _stream_file(file: UploadFile) -> AsyncIterable[bytes]:
    while chunk := await file.read(STREAM_CHUNK_SIZE):
        yield chunk


async def stream_file_to_upload(file: UploadFile) -> ImageEntry:
    first_chunk = await file.read(FLEEP_HEADER_SIZE)
    if not (suffix := _get_image_ext(first_chunk)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is not a valid image",
        )
    file_hash, tmp_path = await _stream_to_temp(_stream_file(file), first_chunk, suffix)
    return await image_index.add(file_hash, tmp_path)


def run_detect(file_path: Path) -> bool:
    """Run two-stage detection.

    Flow:
    1. Stage 1: natural vs screen_like (with TTA)
    2. OOD detection: max_prob < 0.65 → unknown
    3. If natural → return directly
    4. Stage 2: screenshot vs screen_photo
    """
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not available")

    result = predictor.predict(file_path)
    return result["class"] == "screen_photo"
