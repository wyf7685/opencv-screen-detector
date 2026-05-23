"""Image index manager for tracking image IDs and training status."""

import contextlib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import anyio
from pydantic import BaseModel, Field, TypeAdapter

from . import config

UPLOAD_DIR = config.UPLOAD_DIR
INDEX_FILE = UPLOAD_DIR / "index.json"
EXPIRATION_SECONDS = 60 * 10  # 10 mins


class ImageEntry(BaseModel):
    file_name: str
    file_hash: str
    class_name: str | None = None
    trained: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def classified(self) -> bool:
        return self.class_name is not None

    @property
    def path(self) -> Path:
        if self.class_name is None:  # unclassified
            return UPLOAD_DIR / self.file_name
        return UPLOAD_DIR / self.class_name / self.file_name

    def classify(self, class_name: str) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")

        class_dir = UPLOAD_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        new_path = class_dir / self.file_name
        self.path.rename(new_path)

        self.class_name = class_name


class ImageIndex:
    def __init__(self) -> None:
        self._ta = TypeAdapter(dict[str, ImageEntry])
        self._lock = anyio.Lock()
        self._index_file = anyio.Path(INDEX_FILE)

    @contextlib.asynccontextmanager
    async def load_index(self) -> AsyncGenerator[dict[str, ImageEntry]]:
        async with self._lock:
            if await self._index_file.exists():
                index = self._ta.validate_json(await self._index_file.read_bytes())
            else:
                index = {}

            yield index

            await self._index_file.write_bytes(self._ta.dump_json(index))

    async def add(self, file_hash: str, path: Path) -> ImageEntry:
        file_name = f"{file_hash}{path.suffix}"
        new_path = UPLOAD_DIR / file_name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        await anyio.Path(path).rename(new_path)

        entry = ImageEntry(file_name=file_name, file_hash=file_hash)
        async with self.load_index() as index:
            index[file_hash] = entry
        return entry

    async def classify(self, file_hash: str, class_name: str) -> None:
        async with self.load_index() as index:
            if entry := index.get(file_hash):
                entry.classify(class_name)
            else:
                raise ValueError(f"Image not found: {file_hash}")

    async def clean_expired(self) -> None:
        now = datetime.now(UTC)

        async with self.load_index() as index, anyio.create_task_group() as tg:
            expired_hash = {
                file_hash
                for file_hash, entry in index.items()
                if (now - entry.created_at).total_seconds() > EXPIRATION_SECONDS
                and not entry.classified
            }
            for file_hash in expired_hash:
                entry = index.pop(file_hash)
                path = anyio.Path(entry.path)
                if await path.is_file():
                    tg.start_soon(path.unlink)


image_index = ImageIndex()


def batch_mark_trained(*hashes: str) -> int:
    ta = TypeAdapter(dict[str, ImageEntry])
    index = ta.validate_json(INDEX_FILE.read_bytes())
    marked = 0
    for file_hash in hashes:
        if entry := index.get(file_hash):
            entry.trained = True
            marked += 1
    INDEX_FILE.write_bytes(ta.dump_json(index))
    return marked
