from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.types import DateTime
from datetime import datetime

from app.core.database import Base # Importa a Base que definimos

class DetectionResult(Base):
    """
    Modelo SQLAlchemy para a tabela de resultados de detecção.
    Armazenará os dados de cada detecção feita.
    """
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True) # ID único para cada detecção no DB
    image_id = Column(String, index=True, nullable=False) # ID extraído do nome da imagem
    image_filename = Column(String, nullable=False) # Nome completo do arquivo da imagem
    detected_class = Column(String, nullable=False) # Classe detectada (domiciliar, volumoso, poda)
    confidence = Column(Float, nullable=False) # Acurácia/confiança da detecção
    processed_image_path = Column(String, nullable=True) # Caminho para a imagem processada (com BBOXs)
    timestamp = Column(DateTime, default=datetime.utcnow) # Data e hora da detecção

    # Opcional: Adicionar representação para facilitar a depuração
    def __repr__(self):
        return (f"<DetectionResult(id={self.id}, image_id='{self.image_id}', "
                f"detected_class='{self.detected_class}', confidence={self.confidence:.2f})>")