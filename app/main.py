# app/main.py
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path

# Importar seus routers/endpoints
from app.api.endpoints import image_comparation
from app.core.database import create_db_tables, get_db

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

# Monta o diretório 'static' para servir arquivos estáticos (CSS, JS, imagens do frontend)
# A URL acessível será /static/<nome_do_arquivo>
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Monta o diretório de imagens processadas para serem acessíveis via URL
# A URL acessível será /processed_images/<nome_do_arquivo>
app.mount("/processed_images", StaticFiles(directory=PROCESSED_IMAGES_DIR), name="processed_images")

# Configura o Jinja2Templates (usado para servir arquivos HTML diretamente)
templates = Jinja2Templates(directory=STATIC_DIR)

# Evento de inicialização do FastAPI
@app.on_event("startup")
async def startup_event():
    # Cria as tabelas do banco de dados na inicialização
    create_db_tables()
    print("Database tables checked/created.")


# --- Rotas para servir as páginas HTML ---

# Rota raiz ("/") - Assume que você quer que a página de upload seja a primeira
@app.get("/", response_class=HTMLResponse, summary="Página inicial (Upload)")
async def read_root():
    """Redireciona para a página HTML principal de upload de imagem."""
    with open(STATIC_DIR / "painel_upload.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Rota para a página de upload
@app.get("/painel_upload", response_class=HTMLResponse, summary="Página de upload")
async def read_upload_page():
    """Retorna a página HTML de upload de imagem."""
    with open(STATIC_DIR / "painel_upload.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Rota para a página de espera
# ATENÇÃO: ROTA AGORA É "/painel-espera" (com hífen)
@app.get("/painel-espera", response_class=HTMLResponse, summary="Página de espera de processamento")
async def read_wait_page():
    """Retorna a página HTML de espera."""
    with open(STATIC_DIR / "painel_espera.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Rota para a página de resultados
# ATENÇÃO: ROTA AGORA É "/painel_resultados" (com 's' no final)
@app.get("/painel_resultados", response_class=HTMLResponse, summary="Página de resultados")
async def read_results_page():
    """Retorna a página HTML de resultados."""
    with open(STATIC_DIR / "painel_resultados.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# Incluir os routers/endpoints da API
app.include_router(image_comparation.router, prefix="/api", tags=["Image Processing"])