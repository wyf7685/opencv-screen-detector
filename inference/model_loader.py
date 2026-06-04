"""ONNX model loading and session management."""

from pathlib import Path

import onnxruntime as ort


def create_session(model_path: Path) -> ort.InferenceSession | None:
    """Load an ONNX model into an InferenceSession.

    Returns None if the model file does not exist.
    """
    if not model_path.exists():
        return None

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(model_path, sess_options, providers=providers)


class ModelLoader:
    """Manages two-stage ONNX model sessions."""

    def __init__(self, stage1_path: Path, stage2_path: Path) -> None:
        self._stage1_session = create_session(stage1_path)
        self._stage2_session = create_session(stage2_path)

    @property
    def stage1_loaded(self) -> bool:
        return self._stage1_session is not None

    @property
    def stage2_loaded(self) -> bool:
        return self._stage2_session is not None

    @property
    def model_loaded(self) -> bool:
        return self.stage1_loaded and self.stage2_loaded

    @property
    def stage1_session(self) -> ort.InferenceSession:
        assert self._stage1_session is not None, "Stage 1 model not loaded"
        return self._stage1_session

    @property
    def stage2_session(self) -> ort.InferenceSession:
        assert self._stage2_session is not None, "Stage 2 model not loaded"
        return self._stage2_session
