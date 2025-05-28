from fastapi import FastAPI
# from app.api.endpoints import image_processing # Comentada esta linha
from app.core.config import PROCESSED_IMAGES_DIR, XLSX_RESULTS_DIR
from fastapi.staticfiles import StaticFiles

#http://192.168.1.20:8000/

app = FastAPI(
    title="API de Detecção de Lixo com YOLO (Sem Processamento Ativo)", # Título ajustado
    description="API RESTful para detectar e classificar tipos de lixo em imagens usando modelos YOLO e exportar para XLSX. O processamento de imagens está desativado no momento.", # Descrição ajustada
    version="1.0.0"
)

# A linha abaixo foi comentada para desativar o carregamento do endpoint de processamento de imagens
# app.include_router(image_processing.router, prefix="/api", tags=["Image Processing"])

# Monta um diretório estático para servir as imagens processadas (ainda pode ser útil para ver placeholders)
# O frontend poderá acessar essas imagens através de /processed_images/nome_da_imagem.jpg
app.mount("/processed_images", StaticFiles(directory=PROCESSED_IMAGES_DIR), name="processed_images")

# Monta um diretório estático para servir os relatórios XLSX (se algum já existir)
# O frontend poderá acessar esses relatórios através de /reports/nome_do_relatorio.xlsx
app.mount("/reports", StaticFiles(directory=XLSX_RESULTS_DIR), name="reports")

@app.get("/")
async def read_root():
    return {"message": "Bem-vindo à API de Detecção de Lixo com YOLO! O processamento de imagens está desativado por enquanto. Acesse /docs para a documentação interativa."}