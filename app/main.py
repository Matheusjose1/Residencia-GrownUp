# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Corrija o nome do arquivo de importação aqui!
from app.api.endpoints import image_comparation # <--- DEVE SER image_comparation
from app.api.endpoints import db_query_router
from app.core.config import PROCESSED_IMAGES_DIR, XLSX_RESULTS_DIR
from app.core.database import create_db_tables

app = FastAPI(
    title="API de Detecção de Lixo com YOLO",
    description="API RESTful para detectar e classificar tipos de lixo em imagens usando modelos YOLO e exportar para XLSX. Incluindo persistência em DB.",
    version="1.0.0"
)

# Inclui os roteadores
# E aqui você usa o roteador do arquivo correto
app.include_router(image_comparation.router, prefix="/api", tags=["Image Processing"]) # <--- DEVE SER image_comparation.router
app.include_router(db_query_router.router, prefix="/api", tags=["Database Queries"])

# Monta diretórios estáticos
app.mount("/processed_images", StaticFiles(directory=PROCESSED_IMAGES_DIR), name="processed_images")
app.mount("/reports", StaticFiles(directory=XLSX_RESULTS_DIR), name="reports")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    create_db_tables()
    print("API iniciada. Tabelas do DB verificadas/criadas. Modelo YOLO carregado (se best.pt presente).")

<<<<<<< HEAD

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
=======
@app.get("/")
async def serve_frontend():
    """Serve a página principal do frontend."""
    return FileResponse("static/index.html")
>>>>>>> parent of 66fe5eb (Integração)
