from sqlalchemy.orm import Session
from typing import List, Optional

from app.models.detection_result import DetectionResult
from app.schemas.detection_schema import DetectionResultCreate # Importe se for mover a lógica de criação para cá

# Função para criar uma nova detecção no banco de dados (já está em endpoints/image_processing.py, mas aqui é o lugar "certo" para uma função CRUD)
def create_detection(db: Session, detection: DetectionResultCreate, processed_image_path: str) -> DetectionResult:
    db_detection = DetectionResult(
        image_id=detection.image_id,
        image_filename=detection.image_filename,
        detected_class=detection.detected_class,
        confidence=detection.confidence,
        processed_image_path=processed_image_path # Este é um campo adicional que vem do processamento
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection

# Função para obter uma detecção por ID
def get_detection(db: Session, detection_id: int) -> Optional[DetectionResult]:
    return db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()

# Função para obter múltiplas detecções, com paginação
def get_detections(db: Session, skip: int = 0, limit: int = 100) -> List[DetectionResult]:
    return db.query(DetectionResult).offset(skip).limit(limit).all()

# Função para obter detecções por ID da imagem
def get_detections_by_image_id(db: Session, image_id: str) -> List[DetectionResult]:
    return db.query(DetectionResult).filter(DetectionResult.image_id == image_id).all()

# Função para obter detecções por classe (tipo de lixo)
def get_detections_by_class(db: Session, detected_class: str, skip: int = 0, limit: int = 100) -> List[DetectionResult]:
    return db.query(DetectionResult).filter(DetectionResult.detected_class == detected_class).offset(skip).limit(limit).all()