"""ONNX model loading and session management."""

import contextlib
import threading
import time
from collections.abc import Generator
from pathlib import Path

import onnxruntime as ort


def _create_session(model_path: Path) -> ort.InferenceSession:
    """Load an ONNX model into an InferenceSession."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(model_path, sess_options, providers=providers)


class ModelSession:
    IDLE_TIMEOUT = 60.0 * 3

    def __init__(self, path: Path, name: str) -> None:
        self._path = path
        self._name = name
        self._file_state: tuple[float, bool] | None = None  # (mtime, healthy)
        self._session: ort.InferenceSession | None = None
        self._session_lock = threading.Lock()

        self._ref_count = 0
        self._ref_count_lock = threading.Lock()
        self._last_use_ended = 0.0
        self._idle_event = threading.Event()
        self._thread = threading.Thread(target=self._idle_session_monitor, daemon=True)
        self._thread.start()

    def _load(self) -> ort.InferenceSession | None:
        with self._session_lock:
            if self._session is not None:
                return self._session

            try:
                self._session = _create_session(self._path)
            except Exception:
                return None
            return self._session

    def is_available(self) -> bool:
        if not self._path.exists():
            return False
        if self._session is not None:
            return True

        mtime = self._path.stat().st_mtime
        if self._file_state is None or self._file_state[0] != mtime:
            healthy = self._load() is not None
            self._file_state = (mtime, healthy)
        return self._file_state[1]

    @contextlib.contextmanager
    def load(self) -> Generator[ort.InferenceSession]:
        if not self.is_available():
            raise RuntimeError(f"{self._name} model not loaded")

        session = self._load()
        if session is None:
            raise RuntimeError(f"Failed to load {self._name} model")

        with self._ref_count_lock:
            self._ref_count += 1
        try:
            yield session
        finally:
            self._last_use_ended = time.monotonic()
            with self._ref_count_lock:
                self._ref_count -= 1
                if self._ref_count == 0:
                    self._idle_event.set()

    def _idle_session_monitor(self) -> None:
        while True:
            self._idle_event.wait()
            self._idle_event.clear()

            deadline = self._last_use_ended + self.IDLE_TIMEOUT
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

            with self._ref_count_lock:
                if self._ref_count > 0:
                    continue
            if time.monotonic() < self._last_use_ended + self.IDLE_TIMEOUT:
                continue

            with self._session_lock:
                if self._session is not None:
                    self._session = None


class ModelLoader:
    """Manages two-stage ONNX model sessions."""

    def __init__(self, stage1_path: Path, stage2_path: Path) -> None:
        self._stage1_session = ModelSession(stage1_path, "Stage 1")
        self._stage2_session = ModelSession(stage2_path, "Stage 2")

    @property
    def stage1_available(self) -> bool:
        return self._stage1_session.is_available()

    @property
    def stage2_available(self) -> bool:
        return self._stage2_session.is_available()

    @property
    def model_available(self) -> bool:
        return self.stage1_available and self.stage2_available

    def get_stage1_session(
        self,
    ) -> contextlib.AbstractContextManager[ort.InferenceSession]:
        return self._stage1_session.load()

    def get_stage2_session(
        self,
    ) -> contextlib.AbstractContextManager[ort.InferenceSession]:
        return self._stage2_session.load()
