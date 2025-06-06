# app/core/database.py

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from datetime import datetime
from pathlib import Path
import json
import traceback  # <--- ADICIONADO: Importa o módulo traceback
from typing import Optional  # <--- ADICIONADO: Importa Optional do módulo typing

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define o caminho para o arquivo do banco de dados SQLite
DATABASE_DIR = PROJECT_ROOT / "data" / "database"
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    total_images = Column(Integer, default=0, nullable=False)
    processed_images = Column(Integer, default=0, nullable=False)
    completed_images = Column(Integer, default=0, nullable=False)
    failed_images = Column(Integer, default=0, nullable=False)
    overall_progress = Column(Float, default=0.0, nullable=False)  # Progresso de 0.0 a 100.0
    overall_status = Column(String, default="pending", nullable=False)
    message = Column(String, nullable=True)  # Mensagens de status do lote
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)

    # Relacionamento com ImageProcessing
    images = relationship("ImageProcessing", back_populates="batch", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BatchProcessing(batch_id='{self.batch_id}', status='{self.overall_status}')>"


class ImageProcessing(Base):
    __tablename__ = "image_processing"
    id = Column(String, primary_key=True, index=True)  # Usamos o UUID como ID
    batch_processing_id = Column(String, ForeignKey("batch_processing.batch_id"), nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)  # Caminho para a imagem RAW temporária
    status = Column(String, default="pending", nullable=False)  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)

    # Relacionamento com BatchProcessing e ImageProcessingResult
    batch = relationship("BatchProcessing", back_populates="images")
    result = relationship("ImageProcessingResult", back_populates="image_processing", uselist=False,
                          cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ImageProcessing(id='{self.id}', filename='{self.original_filename}', status='{self.status}')>"


class ImageProcessingResult(Base):
    __tablename__ = "image_processing_results"
    id = Column(String, primary_key=True, index=True)  # Usamos o UUID como ID
    image_processing_id = Column(String, ForeignKey("image_processing.id"), unique=True, nullable=False)
    detection_data = Column(Text, nullable=True)  # Armazena dados JSON das detecções
    processed_image_path = Column(String, nullable=True)  # Caminho para a imagem processada
    status = Column(String, default="pending", nullable=False)  # completed, failed, etc.
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)

    # Relacionamento com ImageProcessing
    image_processing = relationship("ImageProcessing", back_populates="result")

    def __repr__(self):
        return f"<ImageProcessingResult(id='{self.id}', status='{self.status}')>"


def create_db_and_tables():
    """Cria as tabelas no banco de dados se elas não existirem."""
    print("Tentando criar tabelas no banco de dados...")
    Base.metadata.create_all(bind=engine)
    print("Tabelas criadas ou já existentes.")


# --- Funções de Ajuda para Interação com o Banco de Dados ---

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def create_db_batch_entry(batch_id: str, total_images: int):
    """
    Cria uma nova entrada de lote no banco de dados.
    Esta função gerencia sua própria sessão do DB.
    """
    print(f"DEBUG_DB: create_db_batch_entry - Iniciando para batch_id={batch_id}, total_images={total_images}")
    db = SessionLocal()  # <<--- LINHA CRÍTICA
    print(f"DEBUG_DB: create_db_batch_entry - Tipo de 'db' após SessionLocal(): {type(db)}")
    print(f"DEBUG_DB: create_db_batch_entry - É 'db' uma instância de Session? {isinstance(db, Session)}")

    try:
        new_batch_entry = BatchProcessing(
            batch_id=batch_id,
            total_images=total_images,
            processed_images=0,
            completed_images=0,
            failed_images=0,
            overall_progress=0,
            overall_status="pending",
            message="Lote criado com sucesso."
        )
        db.add(new_batch_entry)
        db.commit()
        db.refresh(new_batch_entry)
        print(f"DEBUG_DB: Lote {batch_id} criado no DB.")
        return new_batch_entry
    except Exception as e:
        print(f"DEBUG_DB: create_db_batch_entry - Exceção capturada: {e}")
        print(f"DEBUG_DB: create_db_batch_entry - Tipo de 'db' antes do rollback: {type(db)}")
        db.rollback()  # Garante que a transação é revertida em caso de erro
        print(f"ERRO: Erro ao criar entrada de lote {batch_id}: {e}")
        raise  # Relaça a exceção
    finally:
        print(f"DEBUG_DB: create_db_batch_entry - Fechando sessão do DB.")
        db.close()


async def create_db_processing_entry(
        processing_id: str, batch_id: str, original_filename: str, file_path: str, db: Session
):
    """
    Cria uma nova entrada de processamento de imagem no banco de dados.
    Recebe uma sessão DB externa.
    """
    try:
        new_processing_entry = ImageProcessing(
            id=processing_id,
            batch_processing_id=batch_id,
            original_filename=original_filename,
            file_path=file_path,
            status="pending"
        )
        db.add(new_processing_entry)

        # Cria uma entrada ImageProcessingResult associada, também com status 'pending'
        # Seu status será atualizado para 'completed' ou 'failed' após o processamento.
        new_result_entry = ImageProcessingResult(
            id=processing_id,  # Usando o mesmo ID da ImageProcessing
            image_processing_id=processing_id,
            status="pending"
        )
        db.add(new_result_entry)

        db.commit()
        db.refresh(new_processing_entry)
        db.refresh(new_result_entry)
        print(f"Entrada de processamento para {original_filename} (ID: {processing_id}) criada no DB.")
        return new_processing_entry
    except Exception as e:
        db.rollback()
        print(f"Erro ao criar entrada de processamento para {original_filename}: {e}")
        raise


async def update_db_processing_status(db: Session, image_id: str, status: str, message: Optional[str] = None,
                                      force_new_session: bool = False):
    """
    Atualiza o status de processamento de uma imagem e o progresso do lote.
    Pode forçar uma nova sessão DB se a sessão atual falhou.
    """
    local_db = None
    if force_new_session:
        local_db = SessionLocal()
        print(f"DEBUG_DB: update_db_processing_status - Forçando nova sessão para {image_id}.")
        db_to_use = local_db
    else:
        db_to_use = db

    try:
        image_entry = db_to_use.query(ImageProcessing).filter(ImageProcessing.id == image_id).first()
        if not image_entry:
            print(f"AVISO_DB: ImageProcessing entry not found for ID: {image_id}")
            return False

        old_status = image_entry.status
        image_entry.status = status
        image_entry.updated_at = datetime.now()

        # Atualiza o status do resultado também se houver uma entrada
        result_entry = db_to_use.query(ImageProcessingResult).filter(
            ImageProcessingResult.image_processing_id == image_id).first()
        if result_entry:
            result_entry.status = status
            result_entry.updated_at = datetime.now()
            db_to_use.add(result_entry)

        # Lógica para atualizar o progresso do lote
        batch_entry = db_to_use.query(BatchProcessing).filter(
            BatchProcessing.batch_id == image_entry.batch_processing_id).first()
        if batch_entry:
            # Recontar com base no status atualizado
            if old_status == "pending" and status == "completed":
                batch_entry.completed_images += 1
                batch_entry.processed_images += 1
            elif old_status == "pending" and status == "failed":
                batch_entry.failed_images += 1
                batch_entry.processed_images += 1  # Imagens falhas também são "processadas" no sentido de não estarem mais pendentes
            elif old_status == "processing" and status == "completed":  # Transição de 'processing' para 'completed'
                batch_entry.completed_images += 1
                # Se 'processed_images' já foi incrementado ao entrar em 'processing', não incrementa de novo.
                # Se não, ou se você quer que 'processed_images' represente o total já avaliado:
                # batch_entry.processed_images += 1
            elif old_status == "processing" and status == "failed":
                batch_entry.failed_images += 1
            # Para outras transições de status, ajuste a lógica de contagem

            batch_entry.overall_progress = (
                                                       batch_entry.processed_images / batch_entry.total_images) * 100 if batch_entry.total_images > 0 else 0

            if batch_entry.processed_images == batch_entry.total_images:
                batch_entry.overall_status = "completed"
                batch_entry.message = "Todos as imagens do lote foram processadas."
            elif status == "failed":
                # Se uma imagem falha, mas o lote ainda tem imagens pendentes, o lote continua "processing"
                # A menos que todas as imagens restantes tenham falhado.
                if batch_entry.processed_images == batch_entry.total_images and batch_entry.failed_images > 0 and batch_entry.completed_images == 0:
                    batch_entry.overall_status = "failed"
                    batch_entry.message = "Todas as imagens do lote falharam."
                elif batch_entry.processed_images == batch_entry.total_images:  # Misto de sucesso e falha
                    batch_entry.overall_status = "completed_with_errors"
                    batch_entry.message = "Lote concluído com algumas imagens falhas."
            else:
                batch_entry.overall_status = "processing"  # Se ainda houver pendentes

            batch_entry.updated_at = datetime.now()
            db_to_use.add(batch_entry)

        db_to_use.add(image_entry)  # Adiciona a entrada da imagem atualizada
        db_to_use.commit()
        db_to_use.refresh(image_entry)
        if result_entry:
            db_to_use.refresh(result_entry)
        if batch_entry:
            db_to_use.refresh(batch_entry)
        print(
            f"DEBUG_DB: Status da imagem {image_id} atualizado para '{status}'. Lote '{image_entry.batch_processing_id}' atualizado.")
        return True
    except Exception as e:
        print(f"ERRO_DB: Falha ao atualizar status para {image_id}: {e}")
        traceback.print_exc()
        if db_to_use.is_active:
            db_to_use.rollback()
        return False
    finally:
        if local_db:  # Fecha a sessão apenas se ela foi criada internamente (force_new_session)
            local_db.close()


async def get_db_processing_status(image_id: str, db: Session):
    """Obtém o status de processamento de uma imagem específica."""
    status_entry = db.query(ImageProcessing).filter(ImageProcessing.id == image_id).first()
    if status_entry:
        return {
            "id": status_entry.id,
            "batch_processing_id": status_entry.batch_processing_id,
            "original_filename": status_entry.original_filename,
            "status": status_entry.status,
            "created_at": status_entry.created_at.isoformat(),
            "updated_at": status_entry.updated_at.isoformat() if status_entry.updated_at else None
        }
    return None


async def get_db_results(result_id: str, db: Session):
    """Obtém os detalhes dos resultados de processamento de uma imagem."""
    result_entry = db.query(ImageProcessingResult).filter(ImageProcessingResult.id == result_id).first()
    if result_entry:
        detection_data_parsed = []
        if result_entry.detection_data:
            try:
                detection_data_parsed = json.loads(result_entry.detection_data)
            except json.JSONDecodeError:
                print(f"AVISO_DB: Dados de detecção corrompidos para result_id {result_id}")

        return {
            "id": result_entry.id,
            "image_processing_id": result_entry.image_processing_id,
            "detection_data": detection_data_parsed,
            "processed_image_path": result_entry.processed_image_path,
            "status": result_entry.status,
            "created_at": result_entry.created_at.isoformat(),
            "updated_at": result_entry.updated_at.isoformat() if result_entry.updated_at else None
        }
    return None


async def get_db_batch_status(batch_id: str, db: Session):
    """Obtém o status geral de um lote de processamento."""
    batch_status = db.query(BatchProcessing).filter(BatchProcessing.batch_id == batch_id).first()
    if batch_status:
        return {
            "batch_id": batch_status.batch_id,
            "total_images": batch_status.total_images,
            "processed_images": batch_status.processed_images,
            "completed_images": batch_status.completed_images,
            "failed_images": batch_status.failed_images,
            "overall_progress": batch_status.overall_progress,
            "overall_status": batch_status.overall_status,
            "message": batch_status.message,
            "created_at": batch_status.created_at.isoformat(),
            "updated_at": batch_status.updated_at.isoformat() if batch_status.updated_at else None
        }
    return None


async def get_db_all_images_for_batch(batch_id: str, db: Session):
    """
    Carrega todas as entradas de ImageProcessing para um dado batch_id,
    incluindo seus ImageProcessingResult relacionados, se existirem.
    """
    try:
        # Carrega as imagens e seus resultados relacionados em uma única consulta
        images = db.query(ImageProcessing).filter(ImageProcessing.batch_processing_id == batch_id).options(
            relationship(ImageProcessing.result)).all()

        # Desanexa os objetos da sessão para que possam ser usados fora da sessão
        # (especialmente útil se você for serializá-los ou passá-los por muitas camadas)
        for img in images:
            if img.result:
                db.expunge(img.result)
            db.expunge(img)
        return images
    except Exception as e:
        print(f"ERRO_DB: Falha ao carregar imagens para o lote {batch_id}: {e}")
        traceback.print_exc()
        return []  # Retorna lista vazia em caso de erro