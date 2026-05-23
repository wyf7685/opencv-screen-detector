"""Image index manager for tracking image IDs and training status."""

import contextlib
import shutil
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import anyio
from pydantic import BaseModel, Field, TypeAdapter

from . import config

INDEX_FILE = config.DATA_DIR / "image_index.json"
UPLOAD_DIR = config.DATA_DIR / "upload"
EXPIRATION_SECONDS = 60 * 10  # 10 mins


class ImageEntry(BaseModel):
    filename: str
    class_name: str | None = None
    trained: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def classified(self) -> bool:
        return self.class_name is not None

    @property
    def path(self) -> Path:
        if self.class_name is None:
            return UPLOAD_DIR / self.filename
        return UPLOAD_DIR / self.class_name / self.filename

    def classify(self, class_name: str) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")

        class_dir = UPLOAD_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        new_path = class_dir / self.filename
        shutil.move(self.path, new_path)

        self.class_name = class_name


class ImageIndex:
    def __init__(self) -> None:
        self._ta = TypeAdapter(dict[str, ImageEntry])
        self._lock = anyio.Lock()
        self._index_file = anyio.Path(INDEX_FILE)

    @contextlib.asynccontextmanager
    async def load_index(self) -> AsyncGenerator[dict[str, ImageEntry]]:
        """Async context manager to load and save index with locking."""
        async with self._lock:
            if await self._index_file.exists():
                index = self._ta.validate_json(await self._index_file.read_bytes())
            else:
                index = {}

            yield index

            await self._index_file.write_bytes(self._ta.dump_json(index))

    async def add(self, image_id: str, path: Path) -> ImageEntry:
        filename = f"{image_id}{path.suffix}"
        new_path = UPLOAD_DIR / filename
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, new_path)

        entry = ImageEntry(filename=filename)
        async with self.load_index() as index:
            index[image_id] = entry
        return entry

    async def classify(self, image_id: str, class_name: str) -> None:
        async with self.load_index() as index:
            if entry := index.get(image_id):
                entry.classify(class_name)
            else:
                raise ValueError(f"Image ID not found: {image_id}")

    async def clean_expired(self) -> None:
        now = datetime.now(UTC)

        async with self.load_index() as index, anyio.create_task_group() as tg:
            expired_ids = {
                image_id
                for image_id, entry in index.items()
                if (now - entry.created_at).total_seconds() > EXPIRATION_SECONDS
                and not entry.classified
            }
            for image_id in expired_ids:
                entry = index.pop(image_id)
                if entry.path.is_file():
                    tg.start_soon(anyio.Path(entry.path).unlink)


image_index = ImageIndex()


def batch_mark_trained(*image_ids: str) -> int:
    ta = TypeAdapter(dict[str, ImageEntry])
    index = ta.validate_json(INDEX_FILE.read_bytes())
    marked = 0
    for image_id in image_ids:
        if entry := index.get(image_id):
            entry.trained = True
            marked += 1
    INDEX_FILE.write_bytes(ta.dump_json(index))
    return marked
