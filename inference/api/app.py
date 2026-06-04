import contextlib
from collections.abc import AsyncIterator

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..scheduler import run_cleanup_loop
from .predictor import shutdown, startup
from .router import router as api_router


@contextlib.asynccontextmanager
async def lifespan(_: object) -> AsyncIterator[None]:
    startup()
    async with anyio.create_task_group() as tg:
        tg.start_soon(run_cleanup_loop)
        try:
            yield
        finally:
            tg.cancel_scope.cancel()
            shutdown()


app = FastAPI(
    title="Screen Detector API",
    description="Screen detector with two-stage CNN + FFT Branch",
    version="3.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
