# app/core/database.py (Este deve ser o seu módulo de DB, com o conteúdo que te dei por último)

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
from pathlib import Path
import json

# Define o diretório base do projeto
# PROJECT_ROOT será a pasta raiz do projeto (onde app/, data/, static/ estão)
# Como este arquivo está em app/core/database.py:
# Path(__file__).resolve().parent -> app/core
# .parent -> app/
# .parent -> ROOT_DO_PROJETO
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define o caminho para o arquivo do banco de dados SQLite
DATABASE_DIR = PROJECT_ROOT / "data" / "database"  # Isso aponta para /data/database na raiz
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'detections.db'}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Modelos do Banco de Dados ---
class BatchProcessing(Base):
    __tablename__ = "batch_processing"
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_images = Column(Integer, nullable=False)
    processed_images = Column(Integer, default=0)
    completed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    overall_progress = Column(Integer, default=0)
    overall_status = Column(String, default="pending")
    message = Column(String, default="Lote em espera.")
    images = relationship("ImageProcessing", back_populates="batch")


class ImageProcessing(Base):
    __tablename__ = "image_processing"
    id = Column(Integer, primary_key=True, index=True)
    processing_id = Column(String, unique=True, index=True, nullable=False)
    batch_processing_id = Column(String, ForeignKey("batch_processing.batch_id"), nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(String, default="pending")
    progress = Column(Integer, default=0)
    message = Column(String, default="Em espera.")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    batch = relationship("BatchProcessing", back_populates="images")
    result = relationship("ImageProcessingResult", back_populates="image_processing", uselist=False)


class ImageProcessingResult(Base):  # Este é o antigo TrashDetectionResult
    __tablename__ = "image_processing_results"
    id = Column(Integer, primary_key=True, index=True)
    image_processing_id = Column(Integer, ForeignKey("image_processing.id"), nullable=False, unique=True)
    processed_image_path = Column(String, nullable=True)
    excel_report_path = Column(String, nullable=True)
    detection_data = Column(Text, nullable=True)
    average_confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_processing = relationship("ImageProcessing", back_populates="result")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_db_and_tables():
    Base.metadata.create_all(bind=engine)
    print(f"Tabelas do banco de dados criadas (se não existiam) em: {DATABASE_URL}")


# As funções CRUD async também devem estar aqui:
# async def create_db_batch_entry(...):
# ... (todo o resto do código das funções CRUD)
async def create_db_batch_entry(batch_id: str, total_images: int):
    db = SessionLocal()
    try:
        new_batch = BatchProcessing(
            batch_id=batch_id,
            total_images=total_images,
            overall_status="pending",
            message="Lote de processamento criado."
        )
        db.add(new_batch)
        db.commit()
        db.refresh(new_batch)
        return new_batch
    finally:
        db.close()


async def get_db_batch_status(batch_id: str):
    db = SessionLocal()
    try:
        batch = db.query(BatchProcessing).filter(BatchProcessing.batch_id == batch_id).first()
        return batch
    finally:
        db.close()


async def update_db_batch_status(
        batch_id: str,
        processed_images: int,
        completed_images: int,
        failed_images: int,
        overall_progress: int,
        overall_status: str,
        message: str
):
    db = SessionLocal()
    try:
        batch = db.query(BatchProcessing).filter(BatchProcessing.batch_id == batch_id).first()
        if batch:
            batch.processed_images = processed_images
            batch.completed_images = completed_images
            batch.failed_images = failed_images
            batch.overall_progress = overall_progress
            batch.overall_status = overall_status
            batch.message = message
            db.commit()
            db.refresh(batch)
            return batch
    finally:
        db.close()


async def create_db_processing_entry(
        processing_id: str,
        original_filename: str,
        file_path: str,  # Caminho absoluto para o arquivo original
        batch_processing_id: str
):
    db = SessionLocal()
    try:
        new_entry = ImageProcessing(
            processing_id=processing_id,
            batch_processing_id=batch_processing_id,
            original_filename=original_filename,
            file_path=file_path,
            status="pending",
            progress=0,
            message="Aguardando processamento."
        )
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        return new_entry
    finally:
        db.close()


async def get_db_processing_status(processing_id: str):
    db = SessionLocal()
    try:
        # Carrega o resultado junto, se existir
        processing = db.query(ImageProcessing).filter(ImageProcessing.processing_id == processing_id).first()
        if processing:
            # Explicitamente carrega o relacionamento 'result' se não foi carregado
            if processing.result:
                db.expunge(processing.result)  # Desanexa para evitar problemas de sessão
            db.expunge(processing)  # Desanexa a imagem também
        return processing
    finally:
        db.close()


async def update_db_processing_status(
        processing_id: str,
        status: str,
        message: str,
        progress: int,
        detection_data: list = None,
        processed_image_path: str = None,  # Caminho absoluto
        excel_report_path: str = None,  # Caminho absoluto
        average_confidence: float = None
):
    db = SessionLocal()
    try:
        processing = db.query(ImageProcessing).filter(ImageProcessing.processing_id == processing_id).first()
        if processing:
            processing.status = status
            processing.message = message
            processing.progress = progress
            processing.updated_at = datetime.utcnow()  # Usar utcnow para consistência

            if status in ["completed", "failed"] and (detection_data is not None or processed_image_path is not None):
                result = db.query(ImageProcessingResult).filter(
                    ImageProcessingResult.image_processing_id == processing.id).first()
                if not result:
                    result = ImageProcessingResult(image_processing_id=processing.id)
                    db.add(result)

                if detection_data is not None:
                    # Converte a lista de dicionários para JSON string para armazenamento
                    result.detection_data = json.dumps(detection_data)
                if processed_image_path is not None:
                    result.processed_image_path = processed_image_path
                if excel_report_path is not None:
                    result.excel_report_path = excel_report_path
                if average_confidence is not None:
                    result.average_confidence = average_confidence

                db.flush()  # Salva as alterações no resultado antes do commit final

            db.commit()
            db.refresh(processing)
            # Retorna o ID do resultado se ele existe e foi criado/atualizado
            return processing.result.id if processing.result else None
        return None
    finally:
        db.close()


async def get_db_results(result_id: int):
    db = SessionLocal()
    try:
        # Carrega o ImageProcessing junto com o resultado para acessar o original_filename
        result = db.query(ImageProcessingResult).filter(ImageProcessingResult.id == result_id).first()
        if result:
            # Carrega o relacionamento image_processing para acessar o original_filename
            processing_record = db.query(ImageProcessing).filter(
                ImageProcessing.id == result.image_processing_id).first()
            if processing_record:
                result.original_filename = processing_record.original_filename

            # Desserializa os dados de detecção
            result.detection_data = json.loads(result.detection_data) if result.detection_data else []

            db.expunge(result)  # Desanexa o objeto para evitar LazyLoadingError após fechar a sessão
            return result
        return None
    finally:
        db.close()


async def get_db_all_images_for_batch(batch_id: str):
    db = SessionLocal()
    try:
        # Carrega as imagens do lote, e para cada imagem, carrega seu resultado se existir
        images = db.query(ImageProcessing).filter(ImageProcessing.batch_processing_id == batch_id).all()
        # Para que o resultado de cada imagem venha junto, se existir
        for img in images:
            if img.result:  # Acessa o relacionamento para carregá-lo
                db.expunge(img.result)  # Desanexa o resultado
            db.expunge(img)  # Desanexa a imagem também
        return images
    finally:
        db.close()