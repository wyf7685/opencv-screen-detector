"""ONNX model loading and session management."""

from pathlib import Path

import onnxruntime as ort


def create_session(model_path: str | Path) -> ort.InferenceSession | None:
    """Load an ONNX model into an InferenceSession.

    Returns None if the model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        return None

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(str(path), sess_options, providers=providers)


class ModelLoader:
    """Manages two-stage ONNX model sessions."""

    def __init__(
        self,
        stage1_path: str | Path,
        stage2_path: str | Path,
    ) -> None:
        self.stage1_path = Path(stage1_path)
        self.stage2_path = Path(stage2_path)
        self.stage1_session: ort.InferenceSession | None = None
        self.stage2_session: ort.InferenceSession | None = None
        self._load()

    def _load(self) -> None:
        """Load both stage models."""
        self.stage1_session = create_session(self.stage1_path)
        self.stage2_session = create_session(self.stage2_path)

    @property
    def stage1_loaded(self) -> bool:
        return self.stage1_session is not None

    @property
    def stage2_loaded(self) -> bool:
        return self.stage2_session is not None

    @property
    def model_loaded(self) -> bool:
        return self.stage1_loaded and self.stage2_loaded
