# app/main.py
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Optional # Importar Optional para parâmetros opcionais

# Importar seus routers/endpoints
from app.api.endpoints import image_comparation

from app.core.database import (
    create_db_and_tables,
    SessionLocal,
)

# Cria a aplicação FastAPI
app = FastAPI(
    title="Análise de Resíduos com IA",
    description="API para upload, processamento e análise de imagens de resíduos utilizando YOLO.",
    version="1.0.0",
)

# Caminho para o diretório 'static' (relativo à raiz do projeto onde main.py é executado)
STATIC_DIR = Path("static")
# Caminho para o diretório onde as imagens processadas serão salvas e servidas
PROCESSED_IMAGES_DIR = Path("data/output/imagens_processadas")

# Certifica que os diretórios necessários existem
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Monta o diretório 'static' para servir arquivos estáticos (frontend)
# A URL acessível será /static/<nome_do_arquivo>
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Monta o diretório de imagens processadas para serem acessíveis via URL
# A URL acessível será /processed_images/<nome_do_arquivo>
app.mount("/processed_images", StaticFiles(directory=PROCESSED_IMAGES_DIR), name="processed_images")

# Configura o Jinja2Templates (usado para servir arquivos HTML diretamente)
templates = Jinja2Templates(directory=STATIC_DIR)

# Inicialização do FastAPI
@app.on_event("startup")
async def startup_event():
    # Cria as tabelas do banco de dados na inicialização
    create_db_and_tables()
    print("Database tables checked/created.")


# --- ROTAS DE SERVIÇO DE PÁGINAS HTML ---

# Rota raiz ("/") - Redireciona para a página de upload
@app.get("/", response_class=HTMLResponse, summary="Página inicial (Upload)")
async def read_root(request: Request):
    """Redireciona para a página HTML principal de upload de imagem."""
    return templates.TemplateResponse("painel_upload.html", {"request": request})

# Rota para a página de upload
@app.get("/painel_upload", response_class=HTMLResponse, summary="Página de upload")
async def read_upload_page(request: Request):
    """Retorna a página HTML de upload de imagem."""
    return templates.TemplateResponse("painel_upload.html", {"request": request})

# Rota para a página de espera (agora com batch_id OPCIONAL)
# Esta rota atenderá tanto "/painel_espera" quanto "/painel_espera?batch_id=..."
@app.get("/painel_espera", response_class=HTMLResponse, summary="Página de espera de processamento")
async def read_wait_page(request: Request, batch_id: Optional[str] = None):
    """Retorna a página HTML de espera, com batch_id opcional."""
    # O valor de batch_id será None se não for fornecido na URL.
    # O JavaScript no frontend (script_espera.js) será responsável por verificar
    # a presença do batch_id e gerenciar o redirecionamento se ele não existir.
    return templates.TemplateResponse("painel_espera.html", {"request": request})


# Rota para a página de resultados
@app.get("/painel_resultados", response_class=HTMLResponse, summary="Página de resultados")
async def read_results_page(request: Request):
    """Retorna a página HTML de resultados."""
    return templates.TemplateResponse("painel_resultados.html", {"request": request})


# Incluir os routers/endpoints da API
app.include_router(image_comparation.router, prefix="/api", tags=["Image Processing"])

# Se você roda o app via `python main.py` diretamente, descomente o bloco abaixo.
# Se você usa `uvicorn app.main:app --reload`, não precisa.
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)