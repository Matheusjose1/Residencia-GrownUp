from fastapi import FastAPI
from app.api.endpoints import image_comparation
from app.api.endpoints import db_query_router # <-- NOVA IMPORTAÇÃO
from app.core.config import PROCESSED_IMAGES_DIR, XLSX_RESULTS_DIR
from app.core.database import create_db_tables
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="API de Detecção de Lixo com YOLO",
    description="API RESTful para detectar e classificar tipos de lixo em imagens usando modelos YOLO e exportar para XLSX. Incluindo persistência em DB.",
    version="1.0.0"
)

# Inclui o roteador com os endpoints de processamento de imagem
app.include_router(image_comparation.router, prefix="/api", tags=["Image Processing"])
# Inclui o roteador com os endpoints de consulta ao banco de dados
app.include_router(db_query_router.router, prefix="/api", tags=["Database Queries"]) # <-- NOVA LINHA

# Monta diretórios estáticos
app.mount("/processed_images", StaticFiles(directory=PROCESSED_IMAGES_DIR), name="processed_images")
app.mount("/reports", StaticFiles(directory=XLSX_RESULTS_DIR), name="reports")

@app.on_event("startup")
async def startup_event():
    create_db_tables()
    print("API iniciada. Tabelas do DB verificadas/criadas. Pronto para carregar o modelo YOLO.")

@app.get("/")
async def read_root():
    return {"message": "Bem-vindo à API de Detecção de Lixo com YOLO! Acesse /docs para a documentação interativa e os endpoints de processamento de imagens e consulta ao DB."}