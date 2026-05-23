"""Screen Detector API - Main entry point."""

import logging

import uvicorn

from inference.api import app

# Configure logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8325)  # noqa: S104
