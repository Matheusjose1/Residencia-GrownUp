import uuid
import os
import shutil
import json
from typing import Dict, List, Any
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, status, Depends
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import pandas as pd
import traceback

from app.core.database import (
    SessionLocal,
    create_db_and_tables,
    get_db,
    ImageProcessingResult,
    BatchProcessing,
    ImageProcessing,
    # Funções assíncronas do DB
    update_db_processing_status,
    get_db_processing_status,
    create_db_processing_entry,
    create_db_batch_entry,
    get_db_results,
    get_db_batch_status,
    get_db_all_images_for_batch
)

# Importar configurações e modelo YOLO
from app.core.config import PROCESSED_IMAGES_DIR, YOLO_CLASSES, model_yolo_lixeiras, XLSX_RESULTS_DIR

from sqlalchemy.orm import Session

router = APIRouter()

# Dicionário para armazenar o status do processamento (em memória, para simplificar)
processing_status: Dict[str, Dict] = {}


async def process_image_task(processing_id: str, file_path: Path, original_filename: str):
    """
    Função de background para processar a imagem com YOLO e gerar relatório XLSX.
    """
    processing_status[processing_id] = {
        "progress": 0,
        "status": "in_progress",
        "message": "Iniciando processamento...",
        "result_id": None
    }
    await update_db_processing_status(
        processing_id=processing_id,
        status="in_progress",
        message="Iniciando processamento...",
        progress=0
    )

    print(f"[{processing_id}] INFO: Início do processamento da imagem.")

    # Variáveis para armazenar os resultados do processamento
    detected_objects_data: List[Dict[str, Any]] = []
    processed_image_filename: str = None
    final_processed_image_path: Path = None
    excel_filename: str = None
    excel_filepath: Path = None

    yolo_saved_dir: Path = None

    try:
        if not model_yolo_lixeiras:
            raise ValueError("Modelo YOLO não carregado. Não é possível processar a imagem.")
        print(f"[{processing_id}] INFO: Modelo YOLO carregado.")

        processing_status[processing_id]["progress"] = 10
        processing_status[processing_id]["message"] = "Carregando imagem e preparando para detecção..."
        await update_db_processing_status(
            processing_id=processing_id,
            status="in_progress",
            progress=10,
            message="Carregando imagem e preparando para detecção..."
        )

        # Crie um diretório de saída exclusivo para esta execução do YOLO
        yolo_output_base_dir = PROCESSED_IMAGES_DIR / "yolo_runs"
        yolo_run_name = f"run_{processing_id}"
        yolo_saved_dir = yolo_output_base_dir / yolo_run_name
        yolo_saved_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{processing_id}] INFO: Diretório YOLO criado/verificado: {yolo_saved_dir}")

        processing_status[processing_id]["progress"] = 30
        processing_status[processing_id]["message"] = "Executando detecção de objetos..."
        # CORREÇÃO: Ajuste no progress e mensagem para este estágio
        await update_db_processing_status(
            processing_id=processing_id,
            status="in_progress",
            progress=30, # Ajustado para 30
            message="Executando detecção de objetos..." # Mensagem mais apropriada
        )

        print(f"[{processing_id}] INFO: Chamando model_yolo_lixeiras.predict() com source: {file_path}")
        results = model_yolo_lixeiras.predict(
            source=str(file_path),
            save=True,
            conf=0.25,
            iou=0.7,
            project=str(yolo_output_base_dir),
            name=yolo_run_name,
            stream=False,
            verbose=False
        )
        print(
            f"[{processing_id}] INFO: model_yolo_lixeiras.predict() concluído. Resultados: {len(results) if results else 0}")

        processing_status[processing_id]["progress"] = 70
        processing_status[processing_id]["message"] = "Analisando resultados e preparando dados..."
        # CORREÇÃO: Adicionado esta chamada faltante que estava movida para o lugar errado
        await update_db_processing_status(
            processing_id=processing_id,
            status="in_progress",
            progress=70,
            message="Analisando resultados e preparando dados..."
        )

        if results and len(results) > 0:
            result = results[0]
            temp_processed_yolo_image_path = None
            for f in yolo_saved_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    temp_processed_yolo_image_path = f
                    break

            if temp_processed_yolo_image_path:
                processed_image_filename = f"{processing_id}_{original_filename.split('.')[0]}_processed.jpg"
                final_processed_image_path = PROCESSED_IMAGES_DIR / processed_image_filename
                shutil.move(str(temp_processed_yolo_image_path), str(final_processed_image_path))
                print(f"[{processing_id}] INFO: Imagem processada movida para: {final_processed_image_path}")
            else:
                print(
                    f"[{processing_id}] AVISO: Imagem processada pelo YOLO não encontrada em {yolo_saved_dir}. Copiando original.")
                processed_image_filename = f"{processing_id}_{original_filename}"
                final_processed_image_path = PROCESSED_IMAGES_DIR / processed_image_filename
                shutil.copy(str(file_path), str(final_processed_image_path))
                print(f"[{processing_id}] INFO: Imagem original copiada para: {final_processed_image_path}")

            print(f"[{processing_id}] INFO: Extraindo dados de detecção...")
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(val) for val in box.xyxy[0]]
                class_name = YOLO_CLASSES.get(class_id, f"unknown_class_{class_id}")

                detected_objects_data.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox_x1": round(x1, 2),
                    "bbox_y1": round(y1, 2),
                    "bbox_x2": round(x2, 2),
                    "bbox_y2": round(y2, 2)
                })
            print(f"[{processing_id}] INFO: {len(detected_objects_data)} objetos detectados.")

        # --- Geração do XLSX ---
        if detected_objects_data:
            print(f"[{processing_id}] INFO: Iniciando geração do relatório XLSX...")
            try:
                XLSX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(detected_objects_data)
                excel_filename = f"detection_report_{processing_id}.xlsx"
                excel_filepath = XLSX_RESULTS_DIR / excel_filename
                df.to_excel(excel_filepath, index=False, engine='openpyxl')
                print(f"[{processing_id}] INFO: Relatório Excel gerado em: {excel_filepath}")
            except Exception as excel_e:
                print(f"[{processing_id}] ERRO ao gerar relatório Excel: {excel_e}")
                traceback.print_exc()
                excel_filename = None
                excel_filepath = None

        # --- SALVAR NO BANCO DE DADOS ---
        print(f"[{processing_id}] INFO: Salvando resultados no banco de dados...")

        status_message = "Processamento concluído com sucesso!"
        if not processed_image_filename:
            status_message = "Processamento concluído, mas sem imagem YOLO processada."
        if not excel_filename:
            status_message += " Erro ao gerar relatório Excel."

        result_id = await update_db_processing_status(
            processing_id=processing_id,
            status="completed",
            message=status_message,
            progress=100,
            detection_data=detected_objects_data,
            processed_image_path=str(final_processed_image_path) if final_processed_image_path else None,
            excel_report_path=str(excel_filepath) if excel_filepath else None,
        )
        print(f"[{processing_id}] INFO: Resultados salvos no DB com result_id: {result_id}")

    except Exception as e:
        print(f"[{processing_id}] ERRO INESPERADO no processamento da imagem: {e}")
        traceback.print_exc()
        await update_db_processing_status(
            processing_id=processing_id,
            status="failed", # CORREÇÃO: Garantindo que o status seja 'failed' aqui
            message=f"Erro no processamento: {e}",
            progress=0
        )
        processing_status[processing_id]["status"] = "failed"
        processing_status[processing_id]["message"] = f"Erro no processamento: {e}"
        processing_status[processing_id]["progress"] = 0
        print(f"[{processing_id}] ERRO: Status de processamento atualizado para failed.")

    finally:
        if yolo_saved_dir and yolo_saved_dir.exists():
            try:
                shutil.rmtree(str(yolo_saved_dir))
                print(f"[{processing_id}] INFO: Diretório temporário YOLO limpo: {yolo_saved_dir}")
            except OSError as e:
                print(f"[{processing_id}] ERRO: Falha ao remover diretório YOLO temporário {yolo_saved_dir}: {e}")

        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"[{processing_id}] INFO: Arquivo temporário original removido: {file_path}")
            except OSError as e:
                print(f"[{processing_id}] ERRO: Falha ao remover arquivo original {file_path}: {e}")

        current_db_status = await get_db_processing_status(processing_id)
        if current_db_status:
            processing_status[processing_id]["progress"] = current_db_status.progress
            processing_status[processing_id]["status"] = current_db_status.status
            processing_status[processing_id]["message"] = current_db_status.message
            if current_db_status.result:
                processing_status[processing_id]["result_id"] = current_db_status.result.id
                processing_status[processing_id]["processed_image_url"] = (
                    f"/processed_images/{Path(current_db_status.result.processed_image_path).name}"
                    if current_db_status.result and current_db_status.result.processed_image_path else None
                )
                processing_status[processing_id]["excel_report_url"] = (
                    f"/api/download-excel/{Path(current_db_status.result.excel_report_path).name}"
                    if current_db_status.result and current_db_status.result.excel_report_path else None
                )
        print(f"[{processing_id}] INFO: Processamento concluído. Status final: {processing_status[processing_id]}")


@router.post("/upload-image", summary="Faz upload de uma imagem para processamento YOLO")
async def upload_image(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    if not model_yolo_lixeiras:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de detecção de lixo não está carregado. Tente novamente mais tarde."
        )
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nenhuma imagem enviada para processamento."
        )

    batch_id = str(uuid.uuid4())

    await create_db_batch_entry(batch_id=batch_id, total_images=len(files))

    uploaded_files_info = []
    for file in files:
        processing_id = str(uuid.uuid4())
        original_filename = file.filename
        file_extension = Path(original_filename).suffix.lower()

        if file_extension not in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de arquivo não suportado: {original_filename}. Por favor, envie uma imagem (jpg, jpeg, png, gif, bmp, webp)."
            )

        temp_upload_dir = PROCESSED_IMAGES_DIR / "temp_uploads"
        temp_upload_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_upload_dir / f"{processing_id}_{original_filename}"

        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Falha ao salvar o arquivo {original_filename}: {e}")

        await create_db_processing_entry(
            processing_id=processing_id,
            original_filename=original_filename,
            file_path=str(temp_file_path),
            batch_processing_id=batch_id
        )

        background_tasks.add_task(process_image_task, processing_id, temp_file_path, original_filename)
        uploaded_files_info.append({"processing_id": processing_id, "filename": original_filename})

    return JSONResponse({
        "batch_id": batch_id,
        "uploaded_files_info": uploaded_files_info,
        "message": "Upload recebido, processamento iniciado para todos os arquivos."
    })

@router.get("/batch-status/{batch_id}", summary="Obtém o status de processamento de um lote de imagens")
async def get_batch_status(batch_id: str):
    """
    Retorna o status geral de um lote de processamento.
    """
    batch_info = await get_db_batch_status(batch_id)

    if not batch_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lote não encontrado.")

    response_data = {
        "batch_id": batch_info.batch_id,
        "total_images": batch_info.total_images,
        "processed_images": batch_info.processed_images,
        "completed_images": batch_info.completed_images,
        "failed_images": batch_info.failed_images,
        "overall_progress": batch_info.overall_progress,
        "overall_status": batch_info.overall_status,
        "message": batch_info.message,
        "created_at": batch_info.created_at.isoformat() if batch_info.created_at else None,
        "images": []
    }

    return JSONResponse(response_data)

@router.get("/processing-status/{processing_id}", summary="Verifica o status do processamento de uma imagem/vídeo")
async def get_processing_status(processing_id: str):
    """
    Retorna o progresso e o status atual do processamento de uma imagem/vídeo.
    """
    db_status = await get_db_processing_status(processing_id)
    if not db_status:
        status_info = processing_status.get(processing_id)
        if not status_info:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ID de processamento não encontrado.")
        return JSONResponse(status_info)

    response_data = {
        "progress": db_status.progress,
        "status": db_status.status,
        "message": db_status.message,
        "result_id": db_status.result.id if db_status.result else None,
        "processed_image_url": (
            f"/processed_images/{Path(db_status.result.processed_image_path).name}"
            if db_status.result and db_status.result.processed_image_path else None
        ),
        "excel_report_url": (
            f"/api/download-excel/{Path(db_status.result.excel_report_path).name}"
            if db_status.result and db_status.result.excel_report_path else None
        )
    }
    return JSONResponse(response_data)


@router.get("/processing-result/{result_id}", summary="Obtém o resultado do processamento de uma imagem/vídeo")
async def get_processing_result(result_id: int):
    """
    Retorna a imagem processada e os dados de detecção, incluindo URL para download do Excel.
    """
    try:
        db_result = await get_db_results(result_id)

        if not db_result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Resultado não encontrado no banco de dados.")

        processed_image_url = None
        if db_result.processed_image_path:
            processed_image_url = f"/processed_images/{Path(db_result.processed_image_path).name}"

        excel_report_url = None
        if db_result.excel_report_path:
            excel_report_url = f"/api/download-excel/{Path(db_result.excel_report_path).name}"

        original_filename = getattr(db_result, 'original_filename', 'N/A')
        processing_id = getattr(db_result, 'processing_id', 'N/A')

        return JSONResponse({
            "status": "completed",
            "original_filename": original_filename,
            "processing_id": processing_id,
            "processed_image_url": processed_image_url,
            "excel_report_url": excel_report_url,
            "detection_data": db_result.detection_data
        })
    except Exception as e:
        print(f"Erro no endpoint get_processing_result: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro interno do servidor ao obter resultados.")


# --- NOVO ENDPOINT: Rota para download do arquivo XLSX ---
@router.get("/download-excel/{file_name}", summary="Faz download de um arquivo de relatório Excel")
async def download_excel_report(file_name: str):
    """
    Permite o download de um arquivo XLSX de relatório de detecção.
    """
    file_path = XLSX_RESULTS_DIR / file_name

    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Arquivo Excel não encontrado.")

    if not file_path.suffix.lower() == '.xlsx':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tipo de arquivo inválido.")

    return FileResponse(path=file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename=file_name)