import uvicorn
from fastapi import FastAPI

from src.api.detect import router as detect_router

app = FastAPI(title="Screen Detector API")
app.include_router(detect_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8325)  # noqa: S104
