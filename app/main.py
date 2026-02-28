# app/main.py
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Optional
import os

# Importar seus routers/endpoints
from app.api.endpoints import image_comparation

# Importar Base, engine e create_db_and_tables do seu database.py
from app.core.database import (
    create_db_and_tables,
    SessionLocal,
    Base,  # Adicionado: Importar Base para metadata.create_all
    engine # Adicionado: Importar engine para metadata.create_all
)

# Cria a aplicação FastAPI
app = FastAPI(
    title="Análise de Resíduos com IA",
    description="API para upload, processamento e análise de imagens de resíduos utilizando YOLO.",
    version="1.0.0",
)

# Caminho para o diretório 'static' (relativo à raiz do projeto onde main.py é executado)
STATIC_DIR = Path("static")
# Caminho para o diretório onde as imagens processadas serão salvas e servidas (do config.py)
# Isso deve ser consistente com o que está em app/core/config.py
PROCESSED_IMAGES_DIR = Path("data/output/imagens_processadas") # Apenas para garantir que o FastAPI saiba da existência se precisar servir diretamente

# Certifica que os diretórios necessários existem
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True) # Criado também via config.py, mas não custa garantir aqui

# Monta o diretório 'static' para servir arquivos estáticos (frontend: HTML, CSS, JS, imagens)
# A URL acessível será /static/<nome_do_arquivo>
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. O PONTO CHAVE: Servir a pasta 'data' onde as imagens processadas estão
# Sem isso, o navegador nunca conseguirá ler as fotos.
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

# Configura Jinja2Templates para renderizar arquivos HTML
# O diretório 'static' é a raiz onde o Jinja2 vai procurar seus templates HTML
templates = Jinja2Templates(directory=STATIC_DIR)

# --- PARTE CRÍTICA: INICIALIZAÇÃO DO BANCO DE DADOS ---
# Este evento é disparado quando a aplicação FastAPI é iniciada
@app.on_event("startup")
async def startup_event():
    print("Iniciando a aplicação e criando tabelas do banco de dados (se não existirem)...")
    try:
        # Chama a função do database.py que cria as tabelas
        create_db_and_tables()
        print("Tabelas do banco de dados verificadas/criadas com sucesso.")
    except Exception as e:
        print(f"ERRO CRÍTICO durante a inicialização do banco de dados: {e}")
        import traceback
        traceback.print_exc()
        # Se o banco de dados não puder ser inicializado, o aplicativo provavelmente não funcionará.
        # Levantar um erro fatal aqui pode ser útil para depuração em produção.
        raise RuntimeError("Falha ao inicializar o banco de dados. Verifique a conexão e permissões.") from e

# Rota de raiz redireciona para o painel de upload
@app.get("/", response_class=HTMLResponse, summary="Redireciona para o painel de upload")
async def root():
    # Usamos o caminho da rota do FastAPI, não do arquivo estático diretamente.
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/painel_upload">
            <title>Redirecionando...</title>
        </head>
        <body>
            <p>Redirecionando para <a href="/painel_upload">o painel de upload</a>...</p>
        </body>
        </html>
    """, status_code=302)


# Rota para a página de upload
@app.get("/painel_upload", response_class=HTMLResponse, summary="Página de upload de imagem")
async def read_upload_page(request: Request):
    """Retorna a página HTML de upload de imagem."""
    # Apenas o nome do arquivo, pois Jinja2Templates já sabe que o diretório base é 'static'
    return templates.TemplateResponse("painel_upload.html", {"request": request})

# Rota para a página de espera (agora com batch_id OPCIONAL)
@app.get("/painel_espera", response_class=HTMLResponse, summary="Página de espera de processamento")
async def read_wait_page(request: Request, batch_id: Optional[str] = None):
    """Retorna a página HTML de espera, com batch_id opcional."""
    # O valor de batch_id será None se não for fornecido na URL.
    return templates.TemplateResponse("painel_espera.html", {"request": request})


# Rota para a página de resultados
@app.get("/painel_resultado", response_class=HTMLResponse, summary="Página de resultados")
async def read_results_page(request: Request):
    """Retorna a página HTML de resultados."""
    return templates.TemplateResponse("painel_resultado.html", {"request": request})


# Incluir os routers/endpoints da API
app.include_router(image_comparation.router, prefix="/api", tags=["Image Processing"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) # reload=True é ótimo para desenvolvimento