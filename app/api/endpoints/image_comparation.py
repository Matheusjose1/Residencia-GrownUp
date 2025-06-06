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
from PIL import Image # Importar Pillow para manipulação de imagem
from datetime import datetime # Importar datetime para a data da operação

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
# Esta variável não é mais utilizada diretamente, pois o status é gerenciado via DB
# processing_status: Dict[str, Dict] = {}


# Define o threshold de confiança para Verdadeiro Positivo
CONFIDENCE_THRESHOLD_TP = 0.70 # 70%

def classify_detection_by_confidence(confidence: float) -> str:
    """
    Classifica uma detecção com base em seu índice de confiança,
    usando terminologia de métricas de classificação adaptada para a ausência de Ground Truth.
    Esta é uma heurística para categorizar a qualidade da detecção do modelo.
    """
    if confidence >= CONFIDENCE_THRESHOLD_TP:
        return "Verdadeiro Positivo" # Detecção com alta confiança, atende ao critério de 'verdadeiro' do usuário
    else:
        return "Falso Positivo"   # Detecção com baixa confiança, não atinge o critério de 'verdadeiro'


async def process_image_task(processing_id: str, file_path: Path, original_filename: str, db: Session, batch_id: str):
    """
    Função assíncrona para processar uma única imagem usando o modelo YOLO.
    Esta função agora será executada em um pool de threads de segundo plano.
    O argumento 'db' é injetado e a sessão é gerenciada externamente.
    """
    try:
        print(f"[DEBUG] Iniciando processamento para: {original_filename} (ID: {processing_id})")

        if model_yolo_lixeiras is None:
            print("[ERRO] Modelo YOLO não carregado. Verifique config.py.")
            raise ValueError("Modelo YOLO não carregado.")

        print(f"[DEBUG] Executando inferência YOLO para {original_filename}...")
        results = model_yolo_lixeiras(str(file_path))
        print(f"[DEBUG] Inferência YOLO concluída para {original_filename}. Resultados detectados: {len(results)}.")

        detection_data = []
        # Contadores para as classificações baseadas em confiança
        true_positives_count = 0      # Detecções com confiança >= 70%
        false_positives_count = 0     # Detecções com confiança < 70%
        # False Negatives (FN) e True Negatives (TN) não podem ser calculados por imagem
        # sem um ground truth real para comparação.

        image_has_detections = False # Flag para saber se alguma coisa foi detectada

        current_time_for_report = datetime.now() # Data/hora da operação
        operation_date_str = current_time_for_report.strftime("%Y-%m-%d %H:%M:%S")

        if len(results) > 0 and hasattr(results[0], 'boxes'):
            for r in results:
                if hasattr(r, 'boxes') and r.boxes:
                    image_has_detections = True
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        class_name = model_yolo_lixeiras.names[class_id]
                        confidence = float(box.conf[0])
                        # As coordenadas x1, y1, x2, y2 não serão mais usadas no relatório Excel
                        # x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0]]

                        # Usa a nova função para classificar
                        classification_result_type = classify_detection_by_confidence(confidence)

                        if classification_result_type == "Verdadeiro Positivo":
                            true_positives_count += 1
                        elif classification_result_type == "Falso Positivo":
                            false_positives_count += 1

                        detection_data.append({
                            "image_id": processing_id, # Adiciona o ID da imagem processada
                            "class_name": class_name,
                            "confidence": confidence,
                            # "bbox": [x1, y1, x2, y2], # REMOVIDO: Bounding box não será mais exportada para o Excel
                            "operation_date": operation_date_str, # Adiciona a data da operação
                            "classification_type": classification_result_type
                        })
        print(f"[DEBUG] Dados de detecção coletados para {original_filename}: {detection_data}")

        # Resumo das classificações baseadas em confiança
        classification_summary = {
            "Verdadeiro Positivo": true_positives_count,
            "Falso Positivo": false_positives_count,
            "Falso Negativo": 0, # Sem ground truth, não é possível determinar
            "Verdadeiro Negativo": 0, # Sem ground truth/áreas negativas, não é possível determinar
            "Total de Detecções Processadas": true_positives_count + false_positives_count
        }

        # Salvar a imagem processada com as caixas desenhadas
        processed_image_name = f"processed_{processing_id}_{original_filename}"
        processed_image_path = PROCESSED_IMAGES_DIR / processed_image_name

        if image_has_detections and len(results) > 0:
            rendered_image = results[0].plot() # Retorna um array numpy (BGR)
            # Verifica se 'rendered_image' é um array numpy antes de usar Image.fromarray
            if isinstance(rendered_image, (Image.Image, Path)): # Se já é uma Imagem PIL ou Path, não precisa converter
                # Já é um objeto PIL Image ou um path válido, apenas salva
                if isinstance(rendered_image, Image.Image):
                    rendered_image.save(processed_image_path)
                elif isinstance(rendered_image, Path):
                    shutil.copy(rendered_image, processed_image_path)
            else: # Assume que é um array numpy (BGR) e converte para RGB para salvar
                Image.fromarray(rendered_image[..., ::-1]).save(processed_image_path)
            print(f"[DEBUG] Imagem processada com caixas desenhadas salva em: {processed_image_path}")
        else:
            print(f"[DEBUG] Nenhuma detecção para {original_filename}. Copiando imagem original.")
            shutil.copy(file_path, processed_image_path) # Copia a original se não houver detecções

        processed_image_url = f"/static/processed_images/{processed_image_name}"

        # --- Geração do Relatório Excel ---
        # Novo nome do arquivo Excel com data da operação, ID do lote e ID da imagem
        data_operacao_filename = current_time_for_report.strftime("%Y%m%d_%H%M%S")
        excel_file_name = f"relatorio_{data_operacao_filename}_lote_{batch_id}_imagem_{processing_id}.xlsx"
        excel_file_path = XLSX_RESULTS_DIR / excel_file_name
        print(f"[DEBUG] Tentando gerar Excel em: {excel_file_path}")

        excel_report_url = None
        if detection_data:
            try:
                # Criar DataFrame para as detecções individuais
                df_detections = pd.DataFrame(detection_data)
                df_detections.rename(columns={
                    "image_id": "ID da Imagem", # Nova coluna e nome
                    "class_name": "Classe Detectada", # Nome alterado
                    "confidence": "Grau de Confiança", # Nome alterado
                    "operation_date": "Data da Operação", # Nova coluna e nome
                    "classification_type": "Tipo de Classificação" # Manter
                }, inplace=True)
                df_detections["Grau de Confiança"] = (df_detections["Grau de Confiança"] * 100).round(2).astype(str) + "%"
                # REMOVIDO: df_detections["Coordenadas (x1, y1, x2, x2)"] = df_detections["Coordenadas (x1, y1, x2, x2)"].apply(lambda x: f"[{', '.join(map(str, x))}]")


                # Criar DataFrame para o resumo da classificação
                df_summary = pd.DataFrame.from_dict(classification_summary, orient='index', columns=['Valor'])
                df_summary.index.name = "Métrica de Classificação" # Nomeia o índice para clareza

                # Usar ExcelWriter para salvar múltiplos DataFrames em abas diferentes
                with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
                    df_detections.to_excel(writer, sheet_name='Detecções Individuais', index=False)
                    df_summary.to_excel(writer, sheet_name='Resumo da Classificação')

                print(f"[DEBUG] Relatório Excel com múltiplas abas gerado com sucesso em: {excel_file_path}")
                excel_report_url = f"/api/download-excel/{excel_file_name}"
            except Exception as excel_err:
                print(f"[ERRO CRÍTICO] Falha ao salvar o arquivo Excel para {original_filename}: {excel_err}")
                traceback.print_exc()
                excel_file_name = None
                excel_file_path = None
                excel_report_url = None
        else:
            print(f"[DEBUG] Nenhum dado de detecção para {original_filename}. Relatório Excel não será gerado.")
            excel_file_name = None
            excel_file_path = None
            excel_report_url = None

        # Dados a serem salvos no banco de dados para o frontend
        full_detection_data = {
            "detections": detection_data, # Lista de detecções individuais
            "classification_summary_confidence_based": classification_summary, # Resumo das contagens
            "model_precision_average": 0.0 # Placeholder para a demanda 3
        }

        await update_db_processing_status(
            processing_id,
            "completed",
            processed_image_url,
            excel_report_url,
            json.dumps(full_detection_data),
            db
        )
        print(f"[DEBUG] Status do processamento para {processing_id} atualizado para 'completed'.")

    except Exception as e:
        print(f"[ERRO INESPERADO] na process_image_task para {original_filename} (ID: {processing_id}): {e}")
        traceback.print_exc()
        # Garante que o JSON de falha ainda inclua as chaves esperadas
        await update_db_processing_status(
            processing_id,
            "failed",
            processed_image_url=None,
            excel_report_url=None,
            detection_data_json=json.dumps({
                "detections": [],
                "classification_summary_confidence_based": {
                    "Verdadeiro Positivo": 0,
                    "Falso Positivo": 0,
                    "Falso Negativo": 0,
                    "Verdadeiro Negativo": 0,
                    "Total de Detecções Processadas": 0
                },
                "model_precision_average": 0.0
            }),
            db=db
        )
        print(f"[DEBUG] Status do processamento para {processing_id} atualizado para 'failed' devido a erro.")


# Rota para upload de imagens em lote
@router.post("/upload-image", response_model=Dict[str, str], summary="Realiza o upload de imagens para processamento em lote")
async def upload_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Múltiplas imagens para processamento"),
    db: Session = Depends(get_db)
):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nenhum arquivo enviado.")

    batch_id = str(uuid.uuid4())
    await create_db_batch_entry(batch_id, db)
    print(f"[DEBUG] Lote criado com ID: {batch_id}")

    temp_image_dir = Path("data/temp_images")
    temp_image_dir.mkdir(parents=True, exist_ok=True)

    processing_ids = []

    for file in files:
        original_filename = file.filename
        file_extension = Path(original_filename).suffix.lower()
        if file_extension not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Tipo de arquivo não suportado: {file_extension}. Apenas .jpg, .jpeg, .png, .webp são permitidos.")

        processing_id = str(uuid.uuid4())
        unique_filename = f"{processing_id}_{original_filename}"
        file_path = temp_image_dir / unique_filename

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"[DEBUG] Arquivo salvo temporariamente em: {file_path}")

            # Cria a entrada no DB antes de iniciar a tarefa em segundo plano
            await create_db_processing_entry(
                processing_id=processing_id,
                batch_processing_id=batch_id,
                original_filename=original_filename,
                file_path=str(file_path),
                db=db
            )
            processing_ids.append(processing_id)

            # Adiciona a tarefa de processamento à fila de tarefas em segundo plano
            # Passamos a sessão 'db' e o 'batch_id' explicitamente para a tarefa de background
            background_tasks.add_task(process_image_task, processing_id, file_path, original_filename, db, batch_id)
            print(f"[DEBUG] Tarefa de background adicionada para {original_filename} (ID: {processing_id})")

        except Exception as e:
            print(f"[ERRO] Falha ao salvar ou agendar {original_filename}: {e}")
            if processing_id in processing_ids:
                await update_db_processing_status(processing_id, "failed", None, None, json.dumps({
                    "detections": [],
                    "classification_summary_confidence_based": {
                        "Verdadeiro Positivo": 0, "Falso Positivo": 0,
                        "Falso Negativo": 0, "Verdadeiro Negativo": 0,
                        "Total de Detecções Processadas": 0
                    },
                    "model_precision_average": 0.0
                }), db)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao processar {original_filename}: {e}")
        finally:
            file.file.close()

    return JSONResponse(content={"message": "Imagens recebidas e processamento iniciado em segundo plano.", "batch_id": batch_id})


# Rota para obter o status de uma imagem individual
@router.get("/image-status/{processing_id}", response_model=Dict[str, Any], summary="Obtém o status de processamento de uma imagem individual")
async def get_image_status(processing_id: str, db: Session = Depends(get_db)):
    status_entry = await get_db_processing_status(processing_id, db)
    if not status_entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ID de processamento não encontrado.")

    response_data = {
        "processing_id": status_entry.id,
        "original_filename": status_entry.original_filename,
        "status": status_entry.status,
        "processed_image_url": status_entry.processed_image_url,
        "excel_report_url": status_entry.excel_report_url,
        "detection_data": json.loads(status_entry.detection_data) if status_entry.detection_data else {},
        "created_at": status_entry.created_at.isoformat(),
        "updated_at": status_entry.updated_at.isoformat() if status_entry.updated_at else None,
    }
    return response_data


# Rota para obter o status de um lote de imagens
@router.get("/batch-status/{batch_id}", response_model=Dict[str, Any], summary="Obtém o status de processamento de um lote de imagens")
async def get_batch_status(batch_id: str, db: Session = Depends(get_db)):
    batch_status_data = await get_db_batch_status(batch_id, db)
    if not batch_status_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ID do lote não encontrado.")

    return batch_status_data

# Rota para obter todos os resultados de imagens para um lote específico
@router.get("/batch-images/{batch_id}", response_model=List[Dict[str, Any]], summary="Lista todos os resultados de imagens para um dado lote")
async def get_all_images_for_batch(batch_id: str, db: Session = Depends(get_db)):
    images_in_batch = await get_db_all_images_for_batch(batch_id, db)
    if not images_in_batch:
        return []

    results_list = []
    for img_entry in images_in_batch:
        result_data = {
            "processing_id": img_entry.id,
            "original_filename": img_entry.original_filename,
            "status": img_entry.status,
            "processed_image_url": img_entry.result.processed_image_url if img_entry.result else None,
            "excel_report_url": img_entry.result.excel_report_url if img_entry.result else None,
            "detection_data": json.loads(img_entry.result.detection_data) if img_entry.result and img_entry.result.detection_data else {},
            "created_at": img_entry.created_at.isoformat(),
            "updated_at": img_entry.updated_at.isoformat() if img_entry.updated_at else None,
        }
        results_list.append(result_data)
    return results_list


# Rota para baixar o arquivo Excel de um resultado específico
@router.get("/download-excel/{file_name:path}", summary="Baixa o arquivo Excel de um resultado de processamento")
async def download_excel_report(file_name: str):
    file_path = XLSX_RESULTS_DIR / file_name
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Arquivo Excel não encontrado.")

    return FileResponse(path=file_path, filename=file_name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# Rota para baixar um arquivo ZIP de resultados (imagens processadas + Excel + Original)
@router.get("/download-zip/{result_id}", summary="Baixa um arquivo ZIP contendo a imagem processada, original e o relatório Excel (se disponíveis) de um resultado específico.")
async def download_zip_report(result_id: str, db: Session = Depends(get_db)):
    result_entry = db.query(ImageProcessingResult).filter(ImageProcessingResult.id == result_id).first()
    if not result_entry or result_entry.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resultado do processamento não encontrado ou não concluído.")

    processing_record = db.query(ImageProcessing).filter(ImageProcessing.id == result_entry.image_processing_id).first()

    zip_file_name = f"relatorio_{result_id}.zip"
    zip_file_path = XLSX_RESULTS_DIR / zip_file_name

    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Adicionar a imagem processada
            if result_entry.processed_image_url:
                image_name_in_dir = Path(result_entry.processed_image_url).name
                actual_image_path = PROCESSED_IMAGES_DIR / image_name_in_dir
                if actual_image_path.exists():
                    zipf.write(actual_image_path, actual_image_path.name)
                else:
                    print(f"AVISO: Imagem processada não encontrada em: {actual_image_path}")

            # Adicionar o arquivo Excel (com o novo nome)
            if result_entry.excel_report_url:
                excel_name_in_dir = Path(result_entry.excel_report_url).name
                actual_excel_path = XLSX_RESULTS_DIR / excel_name_in_dir
                if actual_excel_path.exists():
                    zipf.write(actual_excel_path, actual_excel_path.name) # Adiciona com o nome completo gerado
                else:
                    print(f"AVISO: Arquivo Excel não encontrado em: {actual_excel_path}")

            # Adicionar a imagem original se disponível
            if processing_record and processing_record.file_path:
                original_image_path = Path(processing_record.file_path)
                if original_image_path.exists():
                    zipf.write(original_image_path, f"original_{original_image_path.name}")
                else:
                    print(f"AVISO: Imagem original não encontrada em: {original_image_path}")
            else:
                print(f"AVISO: Não foi possível encontrar o caminho da imagem original para o result_id {result_id}.")


        return FileResponse(path=zip_file_path, filename=zip_file_name, media_type="application/zip",
                            headers={"Content-Disposition": f"attachment; filename={zip_file_name}"})
    except Exception as e:
        print(f"Erro ao criar arquivo ZIP para result_id {result_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro ao gerar arquivo ZIP.")
    finally:
        pass

# Rota para baixar um arquivo ZIP de um lote completo
@router.get("/download-batch-zip/{batch_id}", summary="Baixa um arquivo ZIP contendo todas as imagens processadas, originais e relatórios Excel de um lote.")
async def download_batch_zip_report(batch_id: str, db: Session = Depends(get_db)):
    batch_entry = db.query(BatchProcessing).filter(BatchProcessing.id == batch_id).first()
    if not batch_entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lote não encontrado.")

    zip_file_name = f"lote_relatorio_{batch_id}.zip"
    zip_file_path = XLSX_RESULTS_DIR / zip_file_name

    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            batch_images = db.query(ImageProcessing).filter(ImageProcessing.batch_processing_id == batch_id).all()
            for img_entry in batch_images:
                result_entry = db.query(ImageProcessingResult).filter(ImageProcessingResult.image_processing_id == img_entry.id).first()

                # Adicionar imagem processada
                if result_entry and result_entry.processed_image_url:
                    image_name_in_dir = Path(result_entry.processed_image_url).name
                    actual_image_path = PROCESSED_IMAGES_DIR / image_name_in_dir
                    if actual_image_path.exists():
                        zipf.write(actual_image_path, f"processed_{actual_image_path.name}")
                    else:
                        print(f"AVISO: Imagem processada não encontrada para {img_entry.original_filename} em {actual_image_path}")

                # Adicionar relatório Excel (com o novo nome)
                if result_entry and result_entry.excel_report_url:
                    excel_name_in_dir = Path(result_entry.excel_report_url).name
                    actual_excel_path = XLSX_RESULTS_DIR / excel_name_in_dir
                    if actual_excel_path.exists():
                        # Usar o nome completo do arquivo Excel para incluí-lo no ZIP do lote
                        zipf.write(actual_excel_path, actual_excel_path.name)
                    else:
                        print(f"AVISO: Arquivo Excel não encontrado para {img_entry.original_filename} em {actual_excel_path}")

                # Adicionar imagem original
                if img_entry.file_path:
                    original_file_path = Path(img_entry.file_path)
                    if original_file_path.exists():
                        zipf.write(original_file_path, f"original_{original_file_path.name}")
                    else:
                        print(f"AVISO: Imagem original não encontrada para {img_entry.original_filename} em {original_file_path}")
                else:
                    print(f"AVISO: Caminho da imagem original não registrado para {img_entry.original_filename}.")

        return FileResponse(path=zip_file_path, filename=zip_file_name, media_type="application/zip",
                            headers={"Content-Disposition": f"attachment; filename={zip_file_name}"})
    except Exception as e:
        print(f"Erro ao criar arquivo ZIP para o lote {batch_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Erro ao gerar arquivo ZIP do lote.")
    finally:
        pass