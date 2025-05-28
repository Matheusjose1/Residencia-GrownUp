from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class DetectionResultBase(BaseModel):
    """Base schema for detection results, used for common fields."""
    image_id: str = Field(..., description="ID extracted from the image filename (e.g., last 4 digits).")
    image_filename: str = Field(..., description="Original filename of the image.")
    detected_class: str = Field(..., description="Class of the detected object (e.g., 'domiciliar', 'volumoso', 'poda').")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the detection (0.0 to 1.0).")
    processed_image_path: Optional[str] = Field(None, description="Path to the image with bounding boxes drawn.")

class DetectionResultCreate(DetectionResultBase):
    """Schema for creating a new detection result (input for DB creation)."""
    pass # Currently, same as base, but allows for future differentiation

class DetectionResultOut(DetectionResultBase):
    """Schema for returning detection results (output from API)."""
    id: int = Field(..., description="Unique database ID for the detection entry.")
    timestamp: datetime = Field(..., description="Timestamp when the detection was recorded in UTC.")

    class Config:
        # This tells Pydantic to read data from ORM models
        # (SQLAlchemy models in this case)
        from_attributes = True # For Pydantic v2+
        # orm_mode = True # For Pydantic v1.x (if you're using an older FastAPI/Pydantic)