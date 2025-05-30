import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session  # <-- IMPORTANTE: Session aqui!

# Define o caminho do banco de dados SQLite
DATABASE_URL = "sqlite:///./app_data.db"  # Nome do arquivo do banco de dados

# Cria a base declarativa para os modelos SQLAlchemy
Base = declarative_base()

# Configura o motor do banco de dados
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}  # Necessário para SQLite com FastAPI
)

# Cria uma sessão do banco de dados
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Definição do modelo para a tabela de resultados de detecção de lixo
class TrashDetectionResult(Base):
    __tablename__ = "trash_detection_results"

    id = Column(Integer, primary_key=True, index=True)
    processing_id = Column(String, unique=True, index=True, nullable=False)  # ID do processamento (UUID)
    original_filename = Column(String, nullable=False)
    processed_filename = Column(String, nullable=True)  # Nome do arquivo da imagem processada

    detection_data = Column(JSON, nullable=False)  # Dados das detecções (bounding boxes, classes, confianças)

    created_at = Column(DateTime, default=func.now())  # Timestamp de criação

    def __repr__(self):
        return f"<TrashDetectionResult(id={self.id}, processing_id='{self.processing_id}', original_filename='{self.original_filename}')>"


# Função para criar as tabelas no banco de dados
def create_db_tables():
    Base.metadata.create_all(bind=engine)
    print("Tabelas do banco de dados criadas (se não existiam).")


# NOVO: Função de dependência para obter uma sessão de banco de dados
# Garanta que a indentação desta função está correta:
# Ela deve estar no mesmo nível que 'create_db_tables' e 'class TrashDetectionResult'.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()