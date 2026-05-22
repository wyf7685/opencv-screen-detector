"""Trainer package bootstrap."""

from __future__ import annotations

import logging
import os
import warnings

# Keep trainer startup quiet in offline or unauthenticated environments.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

warnings.filterwarnings(
    "ignore",
    message=r".*unauthenticated requests to the HF Hub.*",
)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
