# app/api/endpoints/image_comparation.py

import uuid
import os
import shutil
import json
import zipfile
from typing import Dict, List, Any
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, status, Depends
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import pandas as pd
import traceback
from PIL import Image
from datetime import datetime
import torch

from app.core.database import (
    SessionLocal,
    get_db,
    ImageProcessingResult,
    BatchProcessing,
    ImageProcessing,
    update_db_processing_status,
    get_db_processing_status,
    create_db_processing_entry,
    create_db_batch_entry,  # <<-- Mantenha esta importação
    get_db_results,
    get_db_batch_status,
    get_db_all_images_for_batch
)

from app.core.config import PROCESSED_IMAGES_DIR, YOLO_CLASSES, model_yolo_lixeiras, XLSX_RESULTS_DIR

from sqlalchemy.orm import Session

router = APIRouter()

CONFIDENCE_THRESHOLD_TP = 0.70


def resize_image_with_padding(image_path: Path, target_size=(640, 640)):
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        target_width, target_height = target_size

        ratio_w = target_width / original_width
        ratio_h = target_height / original_height
        ratio = min(ratio_w, ratio_h)

        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_img = Image.new("RGB", target_size, (0, 0, 0))

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_img.paste(img, (paste_x, paste_y))

        return new_img


async def process_image_task(processing_id: str, file_path: Path, original_filename: str):
    db = SessionLocal()
    batch_processing_id = None

    try:
        processing_entry = db.query(ImageProcessing).filter(ImageProcessing.id == processing_id).first()
        if not processing_entry:
            raise ValueError(f"Entrada de processamento não encontrada para ID: {processing_id}")
        batch_processing_id = processing_entry.batch_processing_id

        print(f"[DEBUG] Executando inferência YOLO para {original_filename}...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = model_yolo_lixeiras(str(file_path), device=device)

        detection_data = []
        has_detections = False

        results_list = results if isinstance(results, list) else [results]

        for r in results_list:
            if r.boxes is None or len(r.boxes) == 0:
                print(
                    f"[DEBUG_YOLO] r.boxes é None ou vazio para {original_filename}. Pulando detecções para este resultado.")
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            print(
                f"[DEBUG] Inferência YOLO concluída para {original_filename}. Resultados detectados (len(boxes)): {len(boxes)}.")

            if len(boxes) > 0:
                for i in range(len(boxes)):
                    class_id = int(classes[i])
                    class_name = YOLO_CLASSES.get(class_id, f"unknown_class_{class_id}")
                    confidence = float(scores[i])

                    classification_type = "Positivo Verdadeiro" if confidence >= CONFIDENCE_THRESHOLD_TP else "Falso Positivo"

                    detection_data.append({
                        "image_id": processing_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "operation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "classification_type": classification_type
                    })
                    has_detections = True
            else:
                print(f"[DEBUG_YOLO] len(boxes) é 0. Nenhuma detecção válida foi processada.")

        print(f"[DEBUG] Dados de detecção coletados para {original_filename}: {detection_data}")

        processed_image_name = f"processed_{processing_id}_{original_filename}"
        processed_image_path = PROCESSED_IMAGES_DIR / processed_image_name

        if has_detections:
            annotated_img_array = results_list[0].plot(conf=True, labels=True, boxes=True)
            Image.fromarray(annotated_img_array[..., ::-1]).save(processed_image_path)
            print(f"[DEBUG] Imagem processada com caixas desenhadas salva em: {processed_image_path}")
        else:
            print(f"[DEBUG] Nenhuma detecção para {original_filename}. Copiando imagem original.")
            shutil.copy(file_path, processed_image_path)

        try:
            result_entry = db.query(ImageProcessingResult).filter(
                ImageProcessingResult.image_processing_id == processing_id).first()
            if result_entry:
                result_entry.detection_data = json.dumps(detection_data)
                result_entry.processed_image_path = str(processed_image_path)
                result_entry.status = "completed"
                db.add(result_entry)
                db.commit()
                db.refresh(result_entry)
                print(f"[DEBUG] Resultado de processamento atualizado para {processing_id}.")
            else:
                print(f"AVISO: Entrada de resultado para {processing_id} não encontrada. Criando uma nova.")
                new_result_entry = ImageProcessingResult(
                    id=processing_id,
                    image_processing_id=processing_id,
                    detection_data=json.dumps(detection_data),
                    processed_image_path=str(processed_image_path),
                    status="completed",
                    created_at=datetime.now()
                )
                db.add(new_result_entry)
                db.commit()
                db.refresh(new_result_entry)
                print(f"[DEBUG] Nova entrada de resultado criada para {processing_id}.")

            if detection_data:
                excel_file_name = f"relatorio_lote_{batch_processing_id}_imagem_{processing_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                excel_file_path = XLSX_RESULTS_DIR / excel_file_name
                print(f"[DEBUG] Tentando gerar Excel em: {excel_file_path}")

                df = pd.DataFrame(detection_data)
                df['image_name'] = original_filename
                df['batch_id'] = batch_processing_id

                cols = ['batch_id', 'image_name', 'class_name', 'confidence', 'classification_type', 'operation_date',
                        'image_id']
                df = df[cols]

                with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Detecções', index=False)
                    worksheet = writer.sheets['Detecções']
                    for i, col in enumerate(df.columns):
                        max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                        worksheet.set_column(i, i, max_len)
                print(f"[DEBUG] Relatório Excel gerado em: {excel_file_path}")
            else:
                print(f"[DEBUG] Nenhuma detecção para {original_filename}. Relatório Excel não será gerado.")

        except Exception as e:
            db.rollback()
            print(f"[ERRO CRÍTICO] Falha ao salvar resultados no DB ou gerar Excel para {original_filename}: {e}")
            traceback.print_exc()
            update_db_processing_status(db, processing_id, "failed")
            raise

    except Exception as e:
        print(f"[ERRO CRÍTICO] Falha geral ao processar imagem {original_filename}: {e}")
        traceback.print_exc()
        if db.is_active:
            db.rollback()
            db.close()
        # Se a sessão fechou ou falhou, force uma nova para a atualização de status
        update_db_processing_status(None, processing_id, "failed", force_new_session=True)
    finally:
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"[DEBUG] Arquivo temporário RAW removido: {file_path}")
            except OSError as e:
                print(f"AVISO: Não foi possível remover o arquivo temporário RAW {file_path}: {e}")
        if db.is_active:  # Garante que a sessão seja fechada apenas se ainda estiver ativa
            db.close()


# Endpoint para upload de imagens
@router.post("/upload-image")
async def upload_image(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nenhum arquivo enviado.")

    batch_id = str(uuid.uuid4())

    # <<<<<< CORREÇÃO AQUI >>>>>>
    # create_db_batch_entry já gerencia sua própria sessão do DB.
    # Removida a criação e fechamento de sessão local aqui.
    try:
        await create_db_batch_entry(batch_id, len(files))  # <<-- CHAMA SEM PASSAR 'db'
    except Exception as e:
        print(f"Erro ao criar entrada de lote no DB: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro ao iniciar o lote de processamento no banco de dados.")

    uploaded_image_ids = []
    TEMP_RAW_IMAGES_DIR = Path("data") / "temp_images"
    TEMP_RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for file in files:
        processing_id = str(uuid.uuid4())
        original_filename = file.filename
        temp_file_path = TEMP_RAW_IMAGES_DIR / f"{processing_id}_{original_filename}"

        try:
            with open(temp_file_path, "wb") as buffer:
                buffer.write(await file.read())

            # Para criar a entrada da imagem, *esta* função (create_db_processing_entry)
            # espera uma sessão DB. Então, uma nova sessão é aberta e fechada por imagem.
            db_image_entry = SessionLocal()
            try:
                await create_db_processing_entry(processing_id, batch_id, original_filename, str(temp_file_path),
                                                 db_image_entry)
            finally:
                db_image_entry.close()

            uploaded_image_ids.append(processing_id)

            background_tasks.add_task(process_image_task, processing_id, temp_file_path, original_filename)

        except Exception as e:
            print(f"[ERRO] Falha ao salvar ou agendar o processamento de {original_filename}: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Erro ao processar {original_filename}: {e}")

    return JSONResponse(content={"message": "Imagens enviadas e processamento agendado.", "batch_id": batch_id,
                                 "image_ids": uploaded_image_ids})


@router.get("/batch-status/{batch_id}")
async def get_batch_status_endpoint(batch_id: str, db: Session = Depends(get_db)):
    batch_status_data = await get_db_batch_status(batch_id, db)
    if not batch_status_data:
        raise HTTPException(status_code=404, detail="Lote não encontrado.")
    return JSONResponse(content=batch_status_data)


@router.get("/image-status/{processing_id}")
async def get_image_status_endpoint(processing_id: str, db: Session = Depends(get_db)):
    status_data = await get_db_processing_status(processing_id, db)
    if not status_data:
        raise HTTPException(status_code=404, detail="Status de imagem não encontrado.")
    return JSONResponse(content=status_data)


@router.get("/results/{result_id}")
async def get_result_details_endpoint(result_id: str, db: Session = Depends(get_db)):
    result_details = await get_db_results(result_id, db)
    if not result_details:
        raise HTTPException(status_code=404, detail="Resultado não encontrado.")
    return JSONResponse(content=result_details)


@router.get("/batch-images/{batch_id}")
async def get_batch_images_endpoint(batch_id: str, db: Session = Depends(get_db)):
    images_in_batch = await get_db_all_images_for_batch(batch_id, db)
    if not images_in_batch:
        return JSONResponse(content=[], status_code=200)

    response_data = []
    for img in images_in_batch:
        img_dict = {
            "id": img.id,
            "original_filename": img.original_filename,
            "status": img.status,
            "file_path": str(img.file_path) if img.file_path else None,
            "processed_image_path": None,
            "detection_data": []
        }
        if img.result:
            img_dict["processed_image_path"] = img.result.processed_image_path
            try:
                img_dict["detection_data"] = json.loads(img.result.detection_data) if img.result.detection_data else []
            except json.JSONDecodeError:
                img_dict["detection_data"] = []
        response_data.append(img_dict)

    return JSONResponse(content=response_data)


@router.get("/download-processed-image/{processing_id}")
async def download_processed_image(processing_id: str):
    db = SessionLocal()
    try:
        result = db.query(ImageProcessingResult).filter(
            ImageProcessingResult.image_processing_id == processing_id).first()
        if not result or not result.processed_image_path:
            raise HTTPException(status_code=404, detail="Imagem processada não encontrada.")

        file_path = Path(result.processed_image_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Arquivo de imagem processada não encontrado no servidor.")

        original_filename = "processed_image.jpg"
        processing_entry = db.query(ImageProcessing).filter(ImageProcessing.id == processing_id).first()
        if processing_entry:
            original_filename = f"processed_{processing_entry.original_filename}"

        return FileResponse(path=file_path, filename=original_filename, media_type="image/jpeg")
    except Exception as e:
        print(f"Erro ao servir imagem processada {processing_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro ao baixar imagem processada.")
    finally:
        db.close()


@router.get("/download-excel-report/{processing_id}")
async def download_excel_report(processing_id: str):
    db = SessionLocal()
    try:
        processing_entry = db.query(ImageProcessing).filter(ImageProcessing.id == processing_id).first()
        if not processing_entry:
            raise HTTPException(status_code=404, detail="Entrada de processamento não encontrada.")

        batch_id = processing_entry.batch_processing_id

        excel_files = list(XLSX_RESULTS_DIR.glob(f"*lote_{batch_id}_imagem_{processing_id}*.xlsx"))

        if not excel_files:
            raise HTTPException(status_code=404,
                                detail="Relatório Excel não encontrado para esta imagem. Pode não haver detecções ou o arquivo ainda não foi gerado.")

        excel_file_path = excel_files[0]
        excel_file_name = excel_file_path.name

        return FileResponse(path=excel_file_path, filename=excel_file_name,
                            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro ao baixar arquivo Excel para processing_id {processing_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro ao baixar arquivo Excel.")
    finally:
        db.close()


@router.get("/download-batch-zip/{batch_id}")
async def download_batch_zip(batch_id: str, db: Session = Depends(get_db)):
    try:
        zip_file_name = f"lote_{batch_id}_resultados.zip"
        zip_file_output_dir = Path("data") / "output"
        zip_file_output_dir.mkdir(parents=True, exist_ok=True)
        zip_file_path = zip_file_output_dir / zip_file_name

        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            images_in_batch = await get_db_all_images_for_batch(batch_id, db)

            if not images_in_batch:
                raise HTTPException(status_code=404,
                                    detail="Nenhuma imagem encontrada para este lote ou lote não existe.")

            for img_entry in images_in_batch:
                if img_entry.file_path and Path(img_entry.file_path).exists():
                    original_file_path = Path(img_entry.file_path)
                    zipf.write(original_file_path, f"originais/{original_file_path.name}")
                else:
                    print(f"AVISO: Imagem original não encontrada para {img_entry.original_filename}.")

                if img_entry.result and img_entry.result.processed_image_path and Path(
                        img_entry.result.processed_image_path).exists():
                    processed_image_path = Path(img_entry.result.processed_image_path)
                    zipf.write(processed_image_path, f"processadas/{processed_image_path.name}")
                else:
                    print(f"AVISO: Imagem processada não encontrada para {img_entry.original_filename}.")

                excel_files_for_image = list(XLSX_RESULTS_DIR.glob(f"*lote_{batch_id}_imagem_{img_entry.id}*.xlsx"))

                if excel_files_for_image:
                    actual_excel_path = excel_files_for_image[0]
                    if actual_excel_path.exists():
                        zipf.write(actual_excel_path, f"relatorios_excel/{actual_excel_path.name}")
                    else:
                        print(
                            f"AVISO: Arquivo Excel não encontrado para {img_entry.original_filename} em {actual_excel_path}")
                else:
                    print(f"DEBUG: Nenhum relatório Excel encontrado para a imagem {img_entry.original_filename}.")

        if not zip_file_path.exists():
            raise HTTPException(status_code=500, detail="O arquivo ZIP não foi gerado ou está vazio.")

        return FileResponse(path=zip_file_path, filename=zip_file_name, media_type="application/zip",
                            headers={"Content-Disposition": f"attachment; filename={zip_file_name}"})
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Erro ao criar arquivo ZIP para o lote {batch_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro ao gerar arquivo ZIP do lote.")
    finally:
        pass  # Você pode adicionar a lógica de remoção do ZIP aqui se desejar