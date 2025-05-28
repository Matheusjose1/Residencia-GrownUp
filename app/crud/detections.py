from sqlalchemy.orm import Session
from typing import List, Optional

from app.models.detection_result import DetectionResult
from app.schemas.detection_schema import DetectionResultCreate # Importe o schema de criação

def create_detection(db: Session, detection: DetectionResultCreate, processed_image_path: str) -> DetectionResult:
    """
    Cria uma única entrada de detecção no banco de dados.
    """
    db_detection = DetectionResult(
        image_id=detection.image_id,
        image_filename=detection.image_filename,
        detected_class=detection.detected_class,
        confidence=detection.confidence,
        processed_image_path=processed_image_path
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection

def create_multiple_detections(db: Session, detections_data: List[dict]) -> List[DetectionResult]:
    """
    Cria múltiplas entradas de detecção no banco de dados a partir de uma lista de dicionários.
    Cada dicionário deve conter os dados necessários para um DetectionResult.
    """
    db_detections = []
    for data in detections_data:
        db_detection = DetectionResult(
            image_id=data["image_id"],
            image_filename=data["image_filename"],
            detected_class=data["detected_class"],
            confidence=data["confidence"],
            processed_image_path=data.get("processed_image_path") # Usar .get para que seja opcional
        )
        db_detections.append(db_detection)

    if db_detections:
        db.add_all(db_detections)
        db.commit()
        # O refresh individualmente pode ser caro para muitos itens.
        # Uma alternativa seria re-consultar o banco ou aceitar que os objetos
        # não terão os IDs gerados automaticamente até uma nova consulta.
        # Para um número moderado, refresh em loop pode ser ok.
        # Para este caso, vamos manter o refresh para ter os IDs no retorno.
        for db_detection in db_detections:
            db.refresh(db_detection)

    return db_detections


# As funções de leitura (get_detection, get_detections, etc.) permanecem as mesmas
def get_detection(db: Session, detection_id: int) -> Optional[DetectionResult]:
    return db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()

def get_detections(db: Session, skip: int = 0, limit: int = 100) -> List[DetectionResult]:
    return db.query(DetectionResult).offset(skip).limit(limit).all()

def get_detections_by_image_id(db: Session, image_id: str) -> List[DetectionResult]:
    return db.query(DetectionResult).filter(DetectionResult.image_id == image_id).all()

def get_detections_by_class(db: Session, detected_class: str, skip: int = 0, limit: int = 100) -> List[DetectionResult]:
    return db.query(DetectionResult).filter(DetectionResult.detected_class == detected_class).offset(skip).limit(limit).all()