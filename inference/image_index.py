"""Image index manager for tracking image IDs and training status."""

import contextlib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import anyio
from pydantic import BaseModel, Field, TypeAdapter

from . import config

EXPIRATION_SECONDS = 60 * 10  # 10 mins


class ImageEntry(BaseModel):
    file_name: str
    file_hash: str
    class_name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def classified(self) -> bool:
        return self.class_name is not None

    @property
    def path(self) -> Path:
        upload_dir = config.settings.upload_dir
        if self.class_name is None:  # unclassified
            return upload_dir / self.file_name
        return upload_dir / self.class_name / self.file_name

    async def async_classify(self, class_name: str) -> None:
        """Move file to class directory (non-blocking)."""
        src = anyio.Path(self.path)
        if not await src.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")

        upload_dir = config.settings.upload_dir
        class_dir = anyio.Path(upload_dir / class_name)
        await class_dir.mkdir(parents=True, exist_ok=True)
        new_path = upload_dir / class_name / self.file_name
        await src.rename(new_path)

        self.class_name = class_name


class ImageIndex:
    def __init__(self) -> None:
        self._ta = TypeAdapter(dict[str, ImageEntry])
        self._lock = anyio.Lock()

    def _get_index_file(self) -> anyio.Path:
        return anyio.Path(config.settings.index_file)

    @contextlib.asynccontextmanager
    async def load_index(self) -> AsyncGenerator[dict[str, ImageEntry]]:
        async with self._lock:
            index_file = self._get_index_file()
            if await index_file.exists():
                index = self._ta.validate_json(await index_file.read_bytes())
            else:
                index = {}

            yield index

            await index_file.write_bytes(self._ta.dump_json(index))

    async def add(self, file_hash: str, path: Path) -> ImageEntry:
        async with self.load_index() as index:
            if file_hash in index:
                await anyio.Path(path).unlink(missing_ok=True)
                entry = index[file_hash]
                await anyio.Path(entry.path).touch()
                return entry

            upload_dir = config.settings.upload_dir
            new_path = upload_dir / f"{file_hash}{path.suffix}"
            await anyio.Path(new_path.parent).mkdir(parents=True, exist_ok=True)
            await anyio.Path(path).rename(new_path)
            entry = ImageEntry(file_name=new_path.name, file_hash=file_hash)
            index[file_hash] = entry
            return entry

    async def classify(self, file_hash: str, is_screen: bool) -> ImageEntry:
        class_name = "screen_photo" if is_screen else "normal_photo"
        async with self.load_index() as index:
            if entry := index.get(file_hash):
                await entry.async_classify(class_name)
                return entry

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
