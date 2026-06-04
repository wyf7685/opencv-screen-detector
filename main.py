"""Screen Detector API - Main entry point."""

import uvicorn

from inference.api import app
from inference.log import LOGGING_CONFIG

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8325, log_config=LOGGING_CONFIG)  # noqa: S104
