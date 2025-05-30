from sqlalchemy import create_engine, Column, Integer, String, JSON # <--- Certifique-se de ter 'JSON' aqui
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.types import DateTime
from datetime import datetime

from pathlib import Path
import os

# Define o diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Define o caminho para o arquivo do banco de dados SQLite
DATABASE_DIR = BASE_DIR / "data" / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True) # Garante que o diretório exista
DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'detections.db'}"

# Cria a engine do banco de dados. `connect_args` é necessário para SQLite.
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Cria uma sessão de banco de dados. Cada instância de SessionLocal será uma sessão de banco de dados.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base para nossos modelos declarativos. Nossas classes de modelo herdarão desta base.
Base = declarative_base()

# --- CLASSE DE MODELO TrashDetectionResult (PRECISA ESTAR AQUI!) ---
class TrashDetectionResult(Base):
    __tablename__ = "trash_detection_results"

    id = Column(Integer, primary_key=True, index=True)
    processing_id = Column(String, unique=True, index=True, nullable=False)
    original_filename = Column(String, nullable=False)
    processed_filename = Column(String, nullable=True)
    excel_report_filename = Column(String, nullable=True) # Coluna para o nome do arquivo Excel
    detection_data = Column(JSON, nullable=True) # Dados das detecções em formato JSON
    timestamp = Column(DateTime, default=datetime.utcnow) # Adicionando timestamp para registro


# Função para obter uma sessão de banco de dados
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Função para criar as tabelas no banco de dados
def create_db_tables():
    Base.metadata.create_all(bind=engine)
    print("Tabelas do banco de dados criadas (se não existiam).")