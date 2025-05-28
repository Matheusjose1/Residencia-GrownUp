from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.crud import detections as crud_detections # Importação do módulo detections dentro do pacote crud
from app.schemas.detection_schema import DetectionResultOut # Importa o schema de saída

router = APIRouter()

@router.get("/detections/", response_model=List[DetectionResultOut], tags=["Database Queries"])
async def read_all_detections(
    skip: int = Query(0, description="Number of items to skip (for pagination).", ge=0),
    limit: int = Query(100, description="Maximum number of items to return.", le=200),
    db: Session = Depends(get_db)
):
    """
    Retrieve a list of all historical detection results from the database, with pagination.
    """
    all_detections = crud_detections.get_detections(db, skip=skip, limit=limit)
    return all_detections

@router.get("/detections/image/{image_id}", response_model=List[DetectionResultOut], tags=["Database Queries"])
async def read_detections_by_image_id(
    image_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve all detection results associated with a specific image ID.
    """
    image_detections = crud_detections.get_detections_by_image_id(db, image_id=image_id)
    if not image_detections:
        raise HTTPException(status_code=404, detail=f"No detections found for image ID: {image_id}")
    return image_detections

@router.get("/detections/class/{detected_class}", response_model=List[DetectionResultOut], tags=["Database Queries"])
async def read_detections_by_class(
    detected_class: str,
    skip: int = Query(0, description="Number of items to skip (for pagination).", ge=0),
    limit: int = Query(100, description="Maximum number of items to return.", le=200),
    db: Session = Depends(get_db)
):
    """
    Retrieve all detection results for a specific detected class (e.g., 'domiciliar', 'volumoso', 'poda').
    """
    # Garante que a classe passada como parâmetro seja uma das esperadas
    valid_classes = ["domiciliar", "volumoso", "poda", "Nenhuma detecção"] # Inclua "Nenhuma detecção"
    if detected_class not in valid_classes:
        raise HTTPException(status_code=400, detail=f"Invalid detected class. Must be one of: {', '.join(valid_classes)}")

    class_detections = crud_detections.get_detections_by_class(db, detected_class=detected_class, skip=skip, limit=limit)
    if not class_detections:
        raise HTTPException(status_code=404, detail=f"No detections found for class: {detected_class}")
    return class_detections