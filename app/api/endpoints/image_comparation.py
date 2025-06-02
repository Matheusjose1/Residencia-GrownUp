import uuid
import os
import shutil
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
    ImageProcessing
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
    processing_status[processing_id] = {"progress": 0, "status": "in_progress", "message": "Iniciando processamento...",
                                        "result_id": None}

    print(f"[{processing_id}] INFO: Início do processamento da imagem.")
    try:
        if not model_yolo_lixeiras:
            raise ValueError("Modelo YOLO não carregado. Não é possível processar a imagem.")
        print(f"[{processing_id}] INFO: Modelo YOLO carregado.")

        processing_status[processing_id]["progress"] = 10
        processing_status[processing_id]["message"] = "Carregando imagem e preparando para detecção..."

        yolo_output_base_dir = PROCESSED_IMAGES_DIR / "yolo_runs"
        yolo_output_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{processing_id}] INFO: Diretório YOLO criado/verificado: {yolo_output_base_dir}")

        yolo_run_name = f"run_{processing_id}"

        processing_status[processing_id]["progress"] = 30
        processing_status[processing_id]["message"] = "Executando detecção de objetos..."

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
        print(f"[{processing_id}] INFO: model_yolo_lixeiras.predict() concluído. Resultados: {len(results) if results else 0}")


        processing_status[processing_id]["progress"] = 70
        processing_status[processing_id]["message"] = "Analisando resultados e preparando dados..."

        detected_objects_data = []
        processed_image_filename = None

        if results and len(results) > 0:
            result = results[0]
            yolo_saved_dir = Path(result.save_dir)
            print(f"[{processing_id}] INFO: YOLO saved directory: {yolo_saved_dir}")

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
                print(f"[{processing_id}] AVISO: Imagem processada pelo YOLO não encontrada em {yolo_saved_dir}. Salvará original.")
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
        excel_filename = None
        if detected_objects_data:
            print(f"[{processing_id}] INFO: Iniciando geração do relatório XLSX...")
            try:
                XLSX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

                # Verificar se 'bbox' está no dicionário e achatá-lo para o DataFrame
                # Se você decidiu ter bbox_x1 etc. como colunas separadas no DB, certifique-se que detection_data tem esses campos
                # O código atual já faz isso (x1, y1, x2, y2 direto no dicionário), então só precisa do DataFrame.
                df = pd.DataFrame(detected_objects_data)
                excel_filename = f"detection_report_{processing_id}.xlsx"
                excel_filepath = XLSX_RESULTS_DIR / excel_filename
                df.to_excel(excel_filepath, index=False, engine='openpyxl')

                print(f"[{processing_id}] INFO: Relatório Excel gerado em: {excel_filepath}")
            except Exception as excel_e:
                print(f"[{processing_id}] ERRO ao gerar relatório Excel: {excel_e}")
                traceback.print_exc()
                excel_filename = None

        # --- SALVAR NO BANCO DE DADOS (SQLite) ---
        print(f"[{processing_id}] INFO: Salvando resultados no banco de dados...")
        db = SessionLocal()
        result_id = None
        try:
            db_entry = ImageProcessingResult(
                processing_id=processing_id,
                original_filename=original_filename,
                processed_filename=processed_image_filename,
                excel_report_filename=excel_filename, # <--- MANTENHA ISSO!
                detection_data=detected_objects_data
            )
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)
            result_id = db_entry.id
            print(f"[{processing_id}] INFO: Resultados salvos no DB com result_id: {result_id}")
        except Exception as db_e:
            db.rollback()
            print(f"[{processing_id}] ERRO ao salvar no banco de dados: {db_e}")
            traceback.print_exc()
            processing_status[processing_id]["message"] = f"Erro ao salvar resultados: {db_e}"
        finally:
            db.close()

        # Limpar diretório temporário do YOLO para esta run
        if yolo_saved_dir and yolo_saved_dir.exists():
            shutil.rmtree(str(yolo_saved_dir))
            print(f"[{processing_id}] INFO: Diretório temporário YOLO limpo: {yolo_saved_dir}")

        # Atualizar status final
        processing_status[processing_id]["progress"] = 100
        processing_status[processing_id]["status"] = "completed"
        processing_status[processing_id]["message"] = "Processamento concluído com sucesso!"
        processing_status[processing_id]["result_id"] = result_id
        processing_status[processing_id][
            "processed_image_url"] = f"/processed_images/{processed_image_filename}" if processed_image_filename else None
        processing_status[processing_id][
            "excel_report_filename"] = excel_filename # <--- MANTENHA ISSO!
        print(f"[{processing_id}] INFO: Processamento concluído. Status final: {processing_status[processing_id]}")


    except Exception as e:
        print(f"[{processing_id}] ERRO INESPERADO no processamento da imagem: {e}")
        traceback.print_exc()
        processing_status[processing_id]["status"] = "failed"
        processing_status[processing_id]["message"] = f"Erro no processamento: {e}"
        processing_status[processing_id]["progress"] = 0
        print(f"[{processing_id}] ERRO: Status de processamento atualizado para failed.")

    finally:
        if file_path.exists():
            os.remove(file_path)
            print(f"[{processing_id}] INFO: Arquivo temporário original removido: {file_path}")


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

    if file_extension not in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tipo de arquivo não suportado. Por favor, envie uma imagem (jpg, jpeg, png, gif, bmp, webp)."
        )

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
    Retorna a imagem processada e os dados de detecção, incluindo URL para download do Excel.
    """
    try:
        db_entry = db.query(ImageProcessingResult).filter(ImageProcessingResult.id == result_id).first()
        if not db_entry:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Resultado não encontrado no banco de dados.")

        processed_image_url = None
        if db_entry.processed_filename:
            processed_image_path = PROCESSED_IMAGES_DIR / db_entry.processed_filename
            if processed_image_path.exists():
                processed_image_url = f"/processed_images/{db_entry.processed_filename}"

        excel_report_url = None
        if db_entry.excel_report_filename: # <--- AGORA VERIFICA O CAMPO DO DB
            excel_report_url = f"/api/download-excel/{db_entry.excel_report_filename}" # <--- MANTENHA ISSO!

        return JSONResponse({
            "status": "completed",
            "original_filename": db_entry.original_filename,
            "processed_image_url": processed_image_url,
            "excel_report_url": excel_report_url, # <--- MANTENHA ISSO!
            "detection_data": db_entry.detection_data
        })
    except Exception as e:
        print(f"Erro no endpoint get_processing_result: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro interno do servidor: {e}")


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

    return FileResponse(path=file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=file_name)