# app/api/endpoints/image_comparation.py

import uuid
import os
import shutil
import json
import zipfile # Adicionado: Importar a biblioteca zipfile
from typing import Dict, List, Any
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, status, Depends
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import pandas as pd
import traceback

from app.core.database import (
    SessionLocal,
    get_db,
    ImageProcessingResult,
    BatchProcessing,
    ImageProcessing,
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

from sqlalchemy.orm import Session # Mantenha a importação de Session

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
        await update_db_processing_status(
            processing_id=processing_id,
            status="in_progress",
            progress=30,
            message="Executando detecção de objetos..."
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
            status="failed",
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

        # Comentado para depuração, você pode querer manter os arquivos originais
        # para exibição no carrossel. Se os arquivos originais forem movidos
        # para um local permanente no upload, esta remoção pode ser desejável.
        # if file_path.exists():
        #     try:
        #         os.remove(file_path)
        #         print(f"[{processing_id}] INFO: Arquivo temporário original removido: {file_path}")
        #     except OSError as e:
        #         print(f"[{processing_id}] ERRO: Falha ao remover arquivo original {file_path}: {e}")

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
    # O diretório para uploads temporários de imagens originais, montado em main.py
    # PROCESSED_IMAGES_DIR = Path("data/output/imagens_processadas")
    TEMP_UPLOADS_DIR = PROCESSED_IMAGES_DIR / "temp_uploads"
    TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True) # Garante que o diretório exista

    for file in files:
        processing_id = str(uuid.uuid4())
        original_filename = file.filename
        file_extension = Path(original_filename).suffix.lower()

        if file_extension not in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de arquivo não suportado: {original_filename}. Por favor, envie uma imagem (jpg, jpeg, png, gif, bmp, webp)."
            )

        # Salva o arquivo original no diretório temporário de uploads
        temp_file_path = TEMP_UPLOADS_DIR / f"{processing_id}_{original_filename}"

        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Falha ao salvar o arquivo {original_filename}: {e}")

        await create_db_processing_entry(
            processing_id=processing_id,
            original_filename=original_filename,
            file_path=str(temp_file_path), # Salva o caminho onde o arquivo original está
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
async def get_processing_result(result_id: int, db: Session = Depends(get_db)):
    """
    Retorna a imagem processada e os dados de detecção, incluindo URL para download do Excel e URLs das imagens originais.
    """
    try:
        db_result = await get_db_results(result_id, db)

        if not db_result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Resultado não encontrado no banco de dados.")

        processed_image_url = None
        if db_result.processed_image_path:
            processed_image_url = f"/processed_images/{Path(db_result.processed_image_path).name}"

        excel_report_url = None
        if db_result.excel_report_path:
            excel_report_url = f"/api/download-excel/{Path(db_result.excel_report_path).name}"

        # Obtendo o original_filename e processing_id através do relacionamento
        original_filename = "N/A"
        processing_id = "N/A"
        if db_result.image_processing_entry:
            original_filename = db_result.image_processing_entry.original_filename
            processing_id = db_result.image_processing_entry.processing_id

        # --- LÓGICA PARA OBTER AS IMAGENS ORIGINAIS DO LOTE ---
        original_image_urls = []
        # Acessa o batch_id através do relacionamento image_processing_entry
        if db_result.image_processing_entry and db_result.image_processing_entry.batch_processing_id:
            batch_id = db_result.image_processing_entry.batch_processing_id
            batch_images = await get_db_all_images_for_batch(batch_id, db)
            for img_entry in batch_images:
                # O caminho original foi salvo em `file_path` no `create_db_processing_entry`
                # e aponta para o diretório `TEMP_UPLOADS_DIR`.
                # Precisamos construir a URL pública para essa imagem.
                original_image_name = Path(img_entry.file_path).name
                # A URL pública é /temp_uploads/nome_do_arquivo.ext
                original_image_urls.append(f"/temp_uploads/{original_image_name}")
        else:
            print(f"AVISO: Não foi possível obter batch_id para result_id {result_id}. Carrossel de originais pode estar vazio.")


        return JSONResponse({
            "id": result_id,
            "type": "Volumoso", # Exemplo de tipo, ajuste conforme seu modelo
            "date": "04/06/2025", # Exemplo de data, pode ser dinâmico da DB
            "model_accuracy": 0.85, # Exemplo de precisão, pode ser dinâmico da DB
            "status": "completed",
            "original_filename": original_filename,
            "processing_id": processing_id,
            "processed_image_url": processed_image_url,
            "excel_report_url": excel_report_url,
            "detection_data": db_result.detection_data,
            "original_image_urls": original_image_urls, # NOVO: URLs das imagens originais
            "zip_download_url": f"/api/download-zip/{result_id}" # Adicionado para o botão ZIP
        })
    except Exception as e:
        print(f"Erro no endpoint get_processing_result: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro interno do servidor ao obter resultados.")

# --- Rota para download do arquivo XLSX (existente no seu OpenAPI) ---
@router.get("/download-excel/{file_name}", summary="Faz download de um arquivo de relatório Excel")
async def download_excel_report(file_name: str):
    """
    Permite o download de um arquivo XLSX de relatório de detecção.
    """
    file_path = XLSX_RESULTS_DIR / file_name

    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Arquivo Excel não encontrado.")

    if not file_path.suffix.lower() == '.xlsx':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tipo de arquivo inválido. Apenas arquivos .xlsx são suportados para download direto desta rota.")

    return FileResponse(path=file_path, filename=file_name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# --- NOVO ENDPOINT: Rota para download do arquivo ZIP ---
@router.get("/download-zip/{result_id}", summary="Faz download de um arquivo ZIP de resultados", tags=["Image Processing"])
async def download_zip_report(result_id: int, db: Session = Depends(get_db)):
    """
    Permite o download de um arquivo ZIP contendo imagens originais, processadas e relatórios.
    """
    db_result = await get_db_results(result_id, db)

    if not db_result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resultado não encontrado.")

    zip_file_name = f"results_{result_id}.zip"
    # Crie um diretório temporário para o ZIP se ainda não existir
    temp_zip_dir = Path("data/temp_zips") # Ou defina isso em app.core.config
    temp_zip_dir.mkdir(parents=True, exist_ok=True)
    zip_file_path = temp_zip_dir / zip_file_name

    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Adicionar imagem processada
            if db_result.processed_image_path and Path(db_result.processed_image_path).exists():
                zipf.write(db_result.processed_image_path, Path(db_result.processed_image_path).name)

            # Adicionar relatório Excel
            if db_result.excel_report_path and Path(db_result.excel_report_path).exists():
                zipf.write(db_result.excel_report_path, Path(db_result.excel_report_path).name)

            # Adicionar imagens originais associadas ao lote
            if db_result.image_processing_entry and db_result.image_processing_entry.batch_processing_id:
                batch_id = db_result.image_processing_entry.batch_processing_id
                batch_images = await get_db_all_images_for_batch(batch_id, db)
                for img_entry in batch_images:
                    original_file_path = Path(img_entry.file_path) # Caminho salvo no DB
                    if original_file_path.exists():
                        zipf.write(original_file_path, original_file_path.name) # Adiciona ao ZIP com o nome do arquivo
            else:
                print(f"AVISO: Não foi possível encontrar imagens originais para o result_id {result_id} ou seu lote para inclusão no ZIP.")


        return FileResponse(path=zip_file_path, filename=zip_file_name, media_type="application/zip",
                            headers={"Content-Disposition": f"attachment; filename={zip_file_name}"})
    except Exception as e:
        print(f"Erro ao criar arquivo ZIP para result_id {result_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro ao gerar arquivo ZIP.")
    finally:
        # É uma boa prática remover arquivos temporários após o envio.
        # Você pode usar um `try-finally` para garantir a remoção, ou um sistema de limpeza.
        # Por exemplo, pode agendar a remoção do arquivo ZIP após um curto atraso.
        # shutil.rmtree(zip_file_path) # CUIDADO: Isso irá deletar o arquivo ZIP imediatamente.
        pass # Por enquanto, o arquivo permanece no disco para depuração.