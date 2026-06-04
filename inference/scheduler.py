"""Background scheduler for cleaning up expired upload files."""

import anyio

from .image_index import image_index
from .log import logger


async def run_cleanup_loop(interval: int = 60) -> None:
    """Run cleanup task periodically."""
    logger.info(f"Starting cleanup loop with interval {interval} seconds")
    while True:
        try:
            await image_index.clean_expired()
        except Exception:
            logger.exception("Error during cleanup")
        await anyio.sleep(interval)
