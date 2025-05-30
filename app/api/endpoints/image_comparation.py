import uuid
import os
import shutil
from typing import Dict, List, Any
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, status, Depends
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path

# Importar configurações e modelo YOLO
from app.core.config import PROCESSED_IMAGES_DIR, YOLO_CLASSES, model_yolo_lixeiras, XLSX_RESULTS_DIR
from app.core.database import SessionLocal, TrashDetectionResult, create_db_tables, get_db
from sqlalchemy.orm import Session  # Importação para o tipo Session

router = APIRouter()

# Dicionário para armazenar o status do processamento (em memória, para simplificar)
processing_status: Dict[str, Dict] = {}


async def process_image_task(processing_id: str, file_path: Path, original_filename: str):
    """
    Função de background para processar a imagem com YOLO.
    """
    processing_status[processing_id] = {"progress": 0, "status": "in_progress", "message": "Iniciando processamento...",
                                        "result_id": None}

    try:
        if not model_yolo_lixeiras:
            raise ValueError("Modelo YOLO não carregado. Não é possível processar a imagem.")

        processing_status[processing_id]["progress"] = 10
        processing_status[processing_id]["message"] = "Carregando imagem e preparando para detecção..."

        # Caminho para salvar a imagem processada pelo YOLO
        # O YOLO cria uma estrutura de pastas como `project/name/filename.jpg`
        # Vamos usar o `processing_id` para criar uma pasta única para cada run do YOLO
        yolo_output_base_dir = PROCESSED_IMAGES_DIR / "yolo_runs"
        yolo_output_base_dir.mkdir(parents=True, exist_ok=True)  # Garante que o diretório base exista

        # O 'project' define o diretório raiz para os resultados do YOLO (ex: data/output/imagens_processadas/yolo_runs)
        # O 'name' define a subpasta dentro do 'project' para este processamento específico
        yolo_run_name = f"run_{processing_id}"

        processing_status[processing_id]["progress"] = 30
        processing_status[processing_id]["message"] = "Executando detecção de objetos..."

        # Realiza a detecção e salva os resultados.
        # save=True: salva a imagem com bounding boxes.
        # conf: Limiar de confiança para as detecções. Ajuste para sua necessidade.
        # iou: Limiar de Interseção sobre União para Non-Maximum Suppression (NMS). Ajuste.
        # stream=False: Processa a imagem de uma vez.
        # verbose=False: Reduz o log do YOLO para o console.
        results = model_yolo_lixeiras.predict(
            source=str(file_path),
            save=True,
            conf=0.25,  # Ajuste conforme a performance do seu modelo
            iou=0.7,  # Ajuste conforme a necessidade
            project=str(yolo_output_base_dir),
            name=yolo_run_name,
            stream=False,
            verbose=False  # Para um output mais limpo no terminal
        )

        processing_status[processing_id]["progress"] = 70
        processing_status[processing_id]["message"] = "Analisando resultados e preparando dados..."

        detected_objects_data = []  # Lista para armazenar os dados de cada detecção para o DB
        processed_image_filename = None

        # Após a execução do predict, o YOLO salva a imagem em um subdiretório específico.
        # O 'results' objeto contém a informação do save_dir.
        if results and len(results) > 0:
            result = results[0]  # Para uma única imagem, teremos um único objeto de resultado

            # O save_dir é o caminho completo onde o YOLO salvou os resultados (e a imagem).
            # Ex: data/output/imagens_processadas/yolo_runs/run_UUID/
            yolo_saved_dir = Path(result.save_dir)

            # O nome do arquivo original é mantido pelo YOLO na pasta de saída.
            # Precisamos encontrar o arquivo de imagem dentro dessa pasta.
            # Pode ser .jpg, .png, etc.
            temp_processed_yolo_image_path = None
            for f in yolo_saved_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    temp_processed_yolo_image_path = f
                    break

            if temp_processed_yolo_image_path:
                # Gerar um nome único para a imagem processada final
                processed_image_filename = f"{processing_id}_{original_filename.split('.')[0]}_processed.jpg"  # Sempre JPEG para consistência
                final_processed_image_path = PROCESSED_IMAGES_DIR / processed_image_filename

                # Mover a imagem processada pelo YOLO para o nosso diretório final
                shutil.move(str(temp_processed_yolo_image_path), str(final_processed_image_path))
            else:
                print(f"Aviso: Imagem processada pelo YOLO não encontrada em {yolo_saved_dir}. Salvará original.")
                processed_image_filename = f"{processing_id}_{original_filename}"
                final_processed_image_path = PROCESSED_IMAGES_DIR / processed_image_filename
                shutil.copy(str(file_path), str(final_processed_image_path))

            # Extrair dados das detecções
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                # xyxy retorna [x1, y1, x2, y2] - coordenadas do bounding box
                x1, y1, x2, y2 = [float(val) for val in box.xyxy[0]]
                class_name = YOLO_CLASSES.get(class_id, f"unknown_class_{class_id}")

                detected_objects_data.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 4),  # Arredonda para 4 casas decimais
                    "bbox": {
                        "x1": round(x1, 2), "y1": round(y1, 2),  # Arredonda para 2 casas decimais
                        "x2": round(x2, 2), "y2": round(y2, 2)
                    }
                })

        # --- SALVAR NO BANCO DE DADOS (SQLite) ---
        db = SessionLocal()  # Usar SessionLocal diretamente na tarefa de background
        result_id = None
        try:
            db_entry = TrashDetectionResult(
                processing_id=processing_id,
                original_filename=original_filename,
                processed_filename=processed_image_filename,
                detection_data=detected_objects_data  # Dados das detecções no formato JSON
            )
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)
            result_id = db_entry.id
        except Exception as db_e:
            db.rollback()
            print(f"Erro ao salvar no banco de dados para {processing_id}: {db_e}")
            processing_status[processing_id]["message"] = f"Erro ao salvar resultados: {db_e}"
        finally:
            db.close()

        # Limpar diretório temporário do YOLO para esta run
        if yolo_saved_dir and yolo_saved_dir.exists():
            shutil.rmtree(str(yolo_saved_dir))

        # Atualizar status final
        processing_status[processing_id]["progress"] = 100
        processing_status[processing_id]["status"] = "completed"
        processing_status[processing_id]["message"] = "Processamento concluído com sucesso!"
        processing_status[processing_id]["result_id"] = result_id
        processing_status[processing_id][
            "processed_image_url"] = f"/processed_images/{processed_image_filename}" if processed_image_filename else None

    except Exception as e:
        print(f"Erro inesperado no processamento da imagem ({processing_id}): {e}")
        processing_status[processing_id]["status"] = "failed"
        processing_status[processing_id]["message"] = f"Erro no processamento: {e}"
        processing_status[processing_id]["progress"] = 0

    finally:
        # Remover o arquivo original temporário após o processamento, se existir
        if file_path.exists():
            os.remove(file_path)


@router.post("/upload-image", summary="Faz upload de uma imagem para processamento YOLO")
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Recebe um arquivo de imagem, salva e inicia o processamento em segundo plano.
    """
    if not model_yolo_lixeiras:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de detecção de lixo não está carregado. Tente novamente mais tarde."
        )

    processing_id = str(uuid.uuid4())
    original_filename = file.filename
    file_extension = Path(original_filename).suffix.lower()

    # Validação para aceitar SOMENTE imagens
    if file_extension not in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tipo de arquivo não suportado. Por favor, envie uma imagem (jpg, jpeg, png, gif, bmp, webp)."
        )

    # Salva o arquivo original para processamento
    # É bom salvar o original em um local temporário antes de passar para o YOLO
    temp_upload_dir = PROCESSED_IMAGES_DIR / "temp_uploads"
    temp_upload_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_upload_dir / f"{processing_id}_{original_filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Falha ao salvar o arquivo: {e}")

    background_tasks.add_task(process_image_task, processing_id, temp_file_path, original_filename)

    return JSONResponse({"processing_id": processing_id, "message": "Upload recebido, processamento iniciado."})


@router.get("/processing-status/{processing_id}", summary="Verifica o status do processamento de uma imagem/vídeo")
async def get_processing_status(processing_id: str):
    """
    Retorna o progresso e o status atual do processamento de uma imagem/vídeo.
    """
    status_info = processing_status.get(processing_id)
    if not status_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ID de processamento não encontrado.")

    return JSONResponse(status_info)


@router.get("/processing-result/{result_id}", summary="Obtém o resultado do processamento de uma imagem/vídeo")
async def get_processing_result(result_id: int, db: Session = Depends(get_db)):
    """
    Retorna a imagem processada e os dados de detecção.
    """
    try:
        db_entry = db.query(TrashDetectionResult).filter(TrashDetectionResult.id == result_id).first()
        if not db_entry:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Resultado não encontrado no banco de dados.")

        processed_image_url = None
        if db_entry.processed_filename:
            processed_image_path = PROCESSED_IMAGES_DIR / db_entry.processed_filename
            if processed_image_path.exists():
                processed_image_url = f"/processed_images/{db_entry.processed_filename}"

        # Remover a lógica de excel_report_url se não for gerar Excel por enquanto
        excel_report_url = None

        return JSONResponse({
            "status": "completed",
            "original_filename": db_entry.original_filename,
            "processed_image_url": processed_image_url,
            "excel_report_url": excel_report_url,
            "detection_data": db_entry.detection_data
        })
    except Exception as e:
        print(f"Erro no endpoint get_processing_result: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro interno do servidor: {e}")