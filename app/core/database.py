# app/core/database.py

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session # <--- ADICIONE 'Session' AQUI
from datetime import datetime
from pathlib import Path
import json

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
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String, unique=True, index=True, nullable=False)  # Este é o UUID do lote
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    total_images = Column(Integer, default=0, nullable=False)
    processed_images = Column(Integer, default=0, nullable=False)
    completed_images = Column(Integer, default=0, nullable=False)
    failed_images = Column(Integer, default=0, nullable=False)
    overall_progress = Column(Float, default=0.0, nullable=False)
    overall_status = Column(String, default="pending", nullable=False)
    message = Column(String, nullable=True)  # Mensagem pode ser nula

    # One-to-many relationship with ImageProcessing
    images = relationship("ImageProcessing", back_populates="batch", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<BatchProcessing(id={self.id}, batch_id='{self.batch_id}', status='{self.overall_status}')>"


class ImageProcessing(Base):
    __tablename__ = "image_processing"
    id = Column(String, primary_key=True, index=True)  # Usamos UUIDs como IDs de processamento
    batch_processing_id = Column(String, ForeignKey("batch_processing.batch_id"),
                                 nullable=False)  # Foreign Key para o batch_id
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)  # Caminho temporário da imagem original
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relacionamento One-to-one com ImageProcessingResult (o resultado final)
    result = relationship("ImageProcessingResult", back_populates="image_processing", uselist=False,
                          cascade="all, delete-orphan")
    # Relacionamento Many-to-one com BatchProcessing
    batch = relationship("BatchProcessing", back_populates="images")

    def __repr__(self):
        return f"<ImageProcessing(id='{self.id}', filename='{self.original_filename}', status='{self.status}')>"


class ImageProcessingResult(Base):
    __tablename__ = "image_processing_results"
    id = Column(String, primary_key=True, index=True)  # Corresponde ao ImageProcessing.id
    image_processing_id = Column(String, ForeignKey("image_processing.id"), unique=True, nullable=False)
    processed_image_url = Column(String, nullable=True)
    excel_report_url = Column(String, nullable=True)
    detection_data = Column(Text, nullable=True)  # Armazenar JSON como texto
    status = Column(String, default="pending")  # pending, completed, failed
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relacionamento One-to-one com ImageProcessing (a entrada original)
    image_processing = relationship("ImageProcessing", back_populates="result")

    def __repr__(self):
        return f"<ImageProcessingResult(id='{self.id}', status='{self.status}')>"


def create_db_and_tables():
    """Cria as tabelas no banco de dados se elas não existirem."""
    Base.metadata.create_all(bind=engine)
    print("Tabelas do banco de dados verificadas/criadas.")


# Função para obter a sessão do banco de dados
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Funções de Ajuda do Banco de Dados ---

async def create_db_batch_entry(batch_id: str, db: Session):
    """Cria uma nova entrada de lote no banco de dados."""
    try:
        # Aqui, estamos garantindo que todos os campos obrigatórios são explicitamente nomeados
        # e que 'total_images' recebe o valor 0 (ou outro valor inteiro inicial).
        # O 'id' (Primary Key) é autoincremental e não precisa ser passado.

        # --- LINHA DE DEBUG ---
        print(f"DEBUG: Criando BatchProcessing para batch_id={batch_id}, total_images_init=0, db_type={type(db)}")
        # --- FIM LINHA DE DEBUG ---

        new_batch = BatchProcessing(
            batch_id=batch_id,
            created_at=datetime.now(),
            total_images=0,  # <--- ESTE DEVE SER O VALOR CORRETO
            processed_images=0,
            completed_images=0,
            failed_images=0,
            overall_progress=0.0,
            overall_status="pending",
            message="Lote de processamento criado."
        )
        db.add(new_batch)
        db.commit()
        db.refresh(new_batch)
        print(f"DEBUG: Lote {batch_id} criado e persistido com sucesso no DB.")
        return new_batch
    except Exception as e:
        db.rollback()  # Garante que a transação é revertida em caso de erro
        print(f"ERRO: Falha ao criar entrada de lote no DB para batch_id={batch_id}: {e}")
        raise  # Re-lança a exceção para que o chamador possa tratá-la

async def create_db_processing_entry(
        processing_id: str,
        batch_processing_id: str,
        original_filename: str,
        file_path: str,
        db: Session
):
    """Cria uma nova entrada de processamento de imagem no banco de dados."""
    try:
        new_entry = ImageProcessing(
            id=processing_id,
            batch_processing_id=batch_processing_id,
            original_filename=original_filename,
            file_path=file_path,
            status="pending",
            created_at=datetime.now()
        )
        db.add(new_entry)

        # Cria a entrada de resultado correspondente com o mesmo ID
        new_result_entry = ImageProcessingResult(
            id=processing_id,  # Mesmo ID que ImageProcessing
            image_processing_id=processing_id,
            status="pending",
            created_at=datetime.now()
        )
        db.add(new_result_entry)

        db.commit()
        db.refresh(new_entry)
        db.refresh(new_result_entry)

        # Atualiza o total de imagens no lote (se o lote foi criado)
        batch = db.query(BatchProcessing).filter(BatchProcessing.batch_id == batch_processing_id).first()
        if batch:
            batch.total_images += 1
            db.add(batch)  # Marca o lote para atualização
            db.commit()  # Confirma a atualização do lote

        return new_entry
    except Exception as e:
        db.rollback()
        print(f"Erro ao criar entrada de processamento no DB: {e}")
        raise


async def update_db_processing_status(
        processing_id: str,
        status: str,
        processed_image_url: str | None,
        excel_report_url: str | None,
        detection_data_json: str,
        db: Session
):
    """Atualiza o status e os resultados de uma entrada de processamento de imagem e o lote correspondente."""
    try:
        image_processing_entry = db.query(ImageProcessing).filter(ImageProcessing.id == processing_id).first()
        if not image_processing_entry:
            print(f"Entrada ImageProcessing com ID {processing_id} não encontrada para atualização de status.")
            return

        # Atualiza a entrada de processamento principal
        image_processing_entry.status = status
        image_processing_entry.updated_at = datetime.now()
        db.add(image_processing_entry)  # Marca para atualização

        # Atualiza a entrada de resultado correspondente
        image_result_entry = db.query(ImageProcessingResult).filter(
            ImageProcessingResult.image_processing_id == processing_id).first()
        if image_result_entry:
            image_result_entry.status = status
            image_result_entry.processed_image_url = processed_image_url
            image_result_entry.excel_report_url = excel_report_url
            image_result_entry.detection_data = detection_data_json
            image_result_entry.updated_at = datetime.now()
            db.add(image_result_entry)  # Marca para atualização
        else:
            print(f"AVISO: Entrada ImageProcessingResult para {processing_id} não encontrada ao atualizar status.")

        # Atualiza o status do lote
        batch = db.query(BatchProcessing).filter(
            BatchProcessing.batch_id == image_processing_entry.batch_processing_id).first()
        if batch:
            if status == "completed":
                batch.completed_images += 1
                batch.processed_images += 1
            elif status == "failed":
                batch.failed_images += 1
                batch.processed_images += 1

            # Calcula o progresso geral
            if batch.total_images > 0:
                batch.overall_progress = (batch.processed_images / batch.total_images) * 100
            else:
                batch.overall_progress = 0.0

            # Atualiza o status geral do lote
            if batch.processed_images == batch.total_images and batch.total_images > 0:
                if batch.failed_images == 0:
                    batch.overall_status = "completed"
                    batch.message = "Processamento do lote concluído com sucesso."
                else:
                    batch.overall_status = "completed_with_errors"
                    batch.message = f"Processamento do lote concluído com {batch.failed_images} falhas."
            elif batch.processed_images > 0 and batch.processed_images < batch.total_images:
                batch.overall_status = "processing"
                batch.message = f"Processando lote: {batch.processed_images}/{batch.total_images} imagens."

            db.add(batch)  # Marca o lote para atualização
        else:
            print(f"AVISO: Lote com ID {image_processing_entry.batch_processing_id} não encontrado para atualização.")

        db.commit()  # Confirma todas as alterações de uma vez
        print(
            f"DB: Status de {processing_id} e lote {image_processing_entry.batch_processing_id} atualizados para {status}.")
    except Exception as e:
        db.rollback()
        print(f"Erro ao atualizar status de processamento no DB para {processing_id}: {e}")
        raise


async def get_db_processing_status(processing_id: str, db: Session):
    """Obtém o status e resultados de uma imagem processada pelo seu ID."""
    result = db.query(ImageProcessingResult).filter(ImageProcessingResult.id == processing_id).first()
    if result:
        # Carrega o relacionamento image_processing para acessar o original_filename
        processing_record = db.query(ImageProcessing).filter(
            ImageProcessing.id == result.image_processing_id).first()
        if processing_record:
            result.original_filename = processing_record.original_filename

        # Desserializa os dados de detecção
        result.detection_data = json.loads(result.detection_data) if result.detection_data else {}

        db.expunge(result)  # Desanexa o objeto para evitar LazyLoadingError após fechar a sessão
        return result
    return None


async def get_db_results(db: Session):
    """Retorna todos os resultados de processamento de imagens."""
    results = db.query(ImageProcessingResult).all()
    # Para cada resultado, tente carregar o original_filename do ImageProcessing
    for res in results:
        processing_record = db.query(ImageProcessing).filter(ImageProcessing.id == res.image_processing_id).first()
        if processing_record:
            res.original_filename = processing_record.original_filename

        # Desserializa os dados de detecção
        res.detection_data = json.loads(res.detection_data) if res.detection_data else {}
        db.expunge(res)  # Desanexa cada objeto para evitar problemas de sessão

    return results


async def get_db_batch_status(batch_id: str, db: Session):
    """Obtém o status geral de um lote de processamento."""
    batch_status = db.query(BatchProcessing).filter(BatchProcessing.batch_id == batch_id).first()
    if batch_status:
        db.expunge(batch_status)  # Desanexa o objeto
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


async def get_db_all_images_for_batch(batch_id: str, db: Session):  # Agora 'db' é um parâmetro
    """
    Carrega todas as entradas de ImageProcessing para um dado batch_id,
    incluindo seus ImageProcessingResult relacionados, se existirem.
    """
    try:
        images = db.query(ImageProcessing).filter(ImageProcessing.batch_processing_id == batch_id).all()
        for img in images:
            # Acessa o relacionamento 'result' para carregá-lo (se não estiver carregado)
            # e depois desanexa para evitar problemas de sessão fechada.
            if img.result:
                db.expunge(img.result)
            db.expunge(img)
        return images
    except Exception as e:
        print(f"Erro ao obter imagens para o lote {batch_id}: {e}")
        raise  # Re-raise a exceção para que o chamador possa lidar com ela