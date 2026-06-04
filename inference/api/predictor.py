"""Predictor lifecycle management for the API.

Provides lazy singleton with explicit startup/shutdown and failure logging.
"""

import logging

from ..predictor import ScreenDetectorPredictor

logger = logging.getLogger(__name__)

_predictor: ScreenDetectorPredictor | None = None
_load_attempted: bool = False
_load_error: str | None = None


def startup() -> None:
    """Eagerly load the predictor. Called during FastAPI lifespan."""
    global _predictor, _load_attempted, _load_error
    _load_attempted = True
    try:
        _predictor = ScreenDetectorPredictor()
        logger.info("Predictor loaded successfully")
    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Failed to load predictor")


def shutdown() -> None:
    """Release predictor resources. Called during FastAPI lifespan shutdown."""
    global _predictor
    if _predictor is not None:
        _predictor.stage1_session = None
        _predictor.stage2_session = None
        _predictor = None
        logger.info("Predictor released")


def get_predictor() -> ScreenDetectorPredictor | None:
    """Get the predictor singleton.

    Returns None if the model failed to load. Call startup() first during
    FastAPI lifespan to eagerly load and log any failures.
    """
    return _predictor


def is_loaded() -> bool:
    """Check if the predictor is loaded and ready."""
    return _predictor is not None


def load_error() -> str | None:
    """Return the error message if loading failed, else None."""
    return _load_error
