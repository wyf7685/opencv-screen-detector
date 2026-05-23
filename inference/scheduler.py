"""Background scheduler for cleaning up expired upload files."""

import logging

import anyio

from .image_index import image_index

logger = logging.getLogger(__name__)


async def run_cleanup_loop(interval: int = 60) -> None:
    """Run cleanup task periodically."""
    logger.info("Starting cleanup loop with interval %d seconds", interval)
    while True:
        try:
            await image_index.clean_expired()
        except Exception:
            logger.exception("Error during cleanup")
        await anyio.sleep(interval)
