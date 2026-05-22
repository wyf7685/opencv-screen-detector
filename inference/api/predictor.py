from ..predictor import ScreenDetectorPredictor

_predictor: ScreenDetectorPredictor | None = None


def get_predictor() -> ScreenDetectorPredictor | None:
    """Get or create predictor singleton."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = ScreenDetectorPredictor()
        except Exception:
            return None
    return _predictor
