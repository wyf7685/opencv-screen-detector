from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    stage1_model_loaded: bool
    stage2_model_loaded: bool


class DetectRequest(BaseModel):
    """Request model for URL-based detection."""

    url: str


class DetectResponse(BaseModel):
    """Response model for detection."""

    image_id: str
    is_screen: bool


class ClassifyRequest(BaseModel):
    """Request model for updating image classification."""

    image_id: str
    is_screen: bool


class ClassifyResponse(BaseModel):
    """Response model for class update."""

    image_id: str
    is_screen: bool
    class_name: str
