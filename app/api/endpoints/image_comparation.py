from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import os
from pathlib import Path
import shutil
import uuid  # Para gerar IDs únicos para arquivos temporários

from app.core.config import model_yolo_lixeiras, PROCESSED_IMAGES_DIR, XLSX_RESULTS_DIR, YOLO_CLASSES
from app.core.database import get_db
from app.core.utils import extract_id_from_filename, \
    save_detection_to_xlsx  # Certifique-se que esta importação está correta
from app.models.detection_result import DetectionResult  # Ainda precisamos do modelo para o DB
from app.schemas.detection_schema import DetectionResultCreate  # Importe o schema de criação
from app.crud import detections as crud_detections  # Importe as funções CRUD

router = APIRouter()


@router.post("/process_images/", summary="Processar Imagens para Detecção de Lixo", tags=["Image Processing"])
async def process_images(
        files: List[UploadFile] = File(..., description="Múltiplas imagens para processar (JPG ou PNG)."),
        db: Session = Depends(get_db)
):
    """
    Recebe múltiplas imagens, as processa usando o modelo YOLO de lixeiras,
    salva as detecções no banco de dados, gera imagens com bounding boxes
    e cria um relatório XLSX consolidado.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    processed_results = []
    detection_records_to_db = []  # Lista para armazenar dados para o DB
    xlsx_data = []

    # Cria um diretório temporário para os uploads
    temp_upload_dir = Path("/tmp") / f"uploads_{uuid.uuid4()}"
    temp_upload_dir.mkdir(parents=True, exist_ok=True)

    try:
        for uploaded_file in files:
            # Validação do tipo de arquivo
            if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
                raise HTTPException(status_code=400,
                                    detail=f"Tipo de arquivo não suportado: {uploaded_file.filename}. Apenas JPG e PNG são permitidos.")

            # Salvar arquivo temporariamente
            temp_file_path = temp_upload_dir / uploaded_file.filename
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)

            image_id = extract_id_from_filename(uploaded_file.filename)
            detected_classes = []
            confidences = []
            processed_image_local_path = None  # Inicializa como None

            if model_yolo_lixeiras:
                # Realizar detecção
                results = model_yolo_lixeiras(str(temp_file_path))  # YOLO espera string ou Path

                # Para cada imagem, os resultados podem conter múltiplas detecções
                for r in results:
                    # Salva a imagem com as bounding boxes
                    processed_image_output_filename = f"processed_{uploaded_file.filename}"
                    processed_image_save_path = PROCESSED_IMAGES_DIR / processed_image_output_filename
                    r.save(filename=str(processed_image_save_path))
                    processed_image_local_path = str(
                        processed_image_save_path.name)  # Apenas o nome do arquivo para URL

                    if r.boxes and len(r.boxes) > 0:
                        for box in r.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            detected_class = YOLO_CLASSES.get(class_id, "unknown")  # Mapeia ID para nome da classe

                            detected_classes.append(detected_class)
                            confidences.append(confidence)

                            # Adiciona dados para o DB e XLSX
                            detection_data_for_db = {
                                "image_id": image_id,
                                "image_filename": uploaded_file.filename,
                                "detected_class": detected_class,
                                "confidence": confidence,
                                "processed_image_path": processed_image_local_path
                            }
                            detection_records_to_db.append(detection_data_for_db)

                            xlsx_data.append({
                                "ID Imagem": image_id,
                                "Nome do Arquivo": uploaded_file.filename,
                                "Classe Detectada": detected_class,
                                "Confiança": f"{confidence:.2f}"
                            })
                    else:
                        # Caso nenhuma detecção seja encontrada para esta imagem
                        detection_data_for_db = {
                            "image_id": image_id,
                            "image_filename": uploaded_file.filename,
                            "detected_class": "Nenhuma detecção",
                            "confidence": 0.0,
                            "processed_image_path": processed_image_local_path
                            # Pode ser None ou o path da imagem original se preferir
                        }
                        detection_records_to_db.append(detection_data_for_db)
                        xlsx_data.append({
                            "ID Imagem": image_id,
                            "Nome do Arquivo": uploaded_file.filename,
                            "Classe Detectada": "Nenhuma detecção",
                            "Confiança": "0.00"
                        })
            else:
                # Lógica para quando o modelo NÃO está carregado (mesmo caso de antes)
                print("Aviso: Modelo YOLO de lixeiras não carregado. Registrando sem detecção.")
                processed_image_output_filename = f"processed_{uploaded_file.filename}"
                processed_image_save_path = PROCESSED_IMAGES_DIR / processed_image_output_filename

                # Apenas copia a imagem original para o diretório de processadas
                try:
                    shutil.copy(temp_file_path, processed_image_save_path)
                    processed_image_local_path = str(processed_image_save_path.name)
                except Exception as e:
                    print(f"Erro ao copiar imagem original para processadas: {e}")
                    processed_image_local_path = None  # Garante que o path é None em caso de falha

                detection_data_for_db = {
                    "image_id": image_id,
                    "image_filename": uploaded_file.filename,
                    "detected_class": "Modelo não carregado",  # Classe específica para este caso
                    "confidence": 0.0,
                    "processed_image_path": processed_image_local_path
                }
                detection_records_to_db.append(detection_data_for_db)
                xlsx_data.append({
                    "ID Imagem": image_id,
                    "Nome do Arquivo": uploaded_file.filename,
                    "Classe Detectada": "Modelo não carregado",
                    "Confiança": "0.00"
                })

            # Monta o resultado para a resposta JSON
            processed_results.append({
                "filename": uploaded_file.filename,
                "image_id": image_id,
                "detections": detected_classes if detected_classes else ["Nenhuma detecção"],
                "confidences": confidences if confidences else [0.0],
                "processed_image_url": f"/processed_images/{processed_image_local_path}" if processed_image_local_path else None
            })

        # --- NOVA LÓGICA DE PERSISTÊNCIA ---
        # Salvar todas as detecções no banco de dados de uma vez
        # Criamos objetos Pydantic DetectionResultCreate para cada dicionário
        detection_create_schemas = [DetectionResultCreate(**data) for data in detection_records_to_db]

        # A função create_multiple_detections espera uma lista de dicionários.
        # Precisamos passar os dados corretamente, incluindo 'processed_image_path'.
        # O ideal é que detection_create_schemas já contivesse todos os dados.
        # A forma mais simples aqui é passar a lista original de dicionários 'detection_records_to_db'.
        created_db_records = crud_detections.create_multiple_detections(db, detection_records_to_db)
        print(f"Registradas {len(created_db_records)} detecções no banco de dados.")

        # Gerar o relatório XLSX
        xlsx_filename = f"detections_report_{image_id}_{uuid.uuid4().hex[:8]}.xlsx"
        xlsx_file_path = XLSX_RESULTS_DIR / xlsx_filename
        save_detection_to_xlsx(xlsx_data, str(xlsx_file_path))

        response_content = {
            "message": "Imagens processadas com sucesso!",
            "results": processed_results,
            "report_url": f"/reports/{xlsx_filename}"
        }
        return JSONResponse(content=response_content, status_code=200)

    finally:
        # Limpar o diretório temporário
        if temp_upload_dir.exists():
            shutil.rmtree(temp_upload_dir)
            print(f"Diretório temporário limpo: {temp_upload_dir}")