import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from datetime import datetime
from sqlalchemy.orm import Session # Importa a sessão do SQLAlchemy

from app.models.image_processing import process_single_image_yolo
from app.core.utils import save_detection_to_xlsx, extract_id_from_filename
from app.core.config import PROCESSED_IMAGES_DIR, XLSX_RESULTS_DIR
from app.core.database import get_db # Importa a dependência de DB
from app.models.detection_result import DetectionResult # Importa o modelo do DB

router = APIRouter()


@router.post("/process_images/")
async def process_images(
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db)  # Injeta a sessão do banco de dados aqui
) -> JSONResponse:
    """
    Endpoint para processar uma ou múltiplas imagens usando o modelo YOLO.
    Recebe um ou mais arquivos de imagem, salva os resultados no DB,
    e retorna um JSON com o resumo das detecções e um link para o arquivo XLSX de resultados.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    temp_image_paths = []
    all_flat_detections_for_xlsx = []  # Para coletar dados para o XLSX

    # Lista para coletar os dados que serão persistidos no DB, já no formato do modelo
    detections_to_save_to_db: List[DetectionResult] = []

    try:
        for image_file in files:
            if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise HTTPException(status_code=400,
                                    detail=f"Apenas arquivos JPG ou PNG são permitidos. Recebido: {image_file.filename}")

            temp_path = os.path.join("/tmp", image_file.filename)
            os.makedirs("/tmp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(await image_file.read())
            temp_image_paths.append(temp_path)

            detection_result = process_single_image_yolo(temp_path)

            image_id = extract_id_from_filename(detection_result["image_name"]) or "N/A"

            if detection_result["detections"]:
                for det in detection_result["detections"]:
                    # Prepara dados para o XLSX
                    all_flat_detections_for_xlsx.append({
                        "image_id": image_id,
                        "class": det["class"],
                        "confidence": det["confidence"]
                    })

                    # Prepara dados para o banco de dados
                    new_db_detection = DetectionResult(
                        image_id=image_id,
                        image_filename=detection_result["image_name"],
                        detected_class=det["class"],
                        confidence=det["confidence"],
                        processed_image_path=detection_result["processed_image_path"],
                        timestamp=datetime.utcnow()
                    )
                    detections_to_save_to_db.append(new_db_detection)
            else:
                # Caso não haja detecções, ainda adicionar uma linha para o XLSX e DB
                all_flat_detections_for_xlsx.append({
                    "image_id": image_id,
                    "class": "Nenhuma detecção",
                    "confidence": 0.0
                })
                new_db_detection = DetectionResult(
                    image_id=image_id,
                    image_filename=detection_result["image_name"],
                    detected_class="Nenhuma detecção",
                    confidence=0.0,
                    processed_image_path=detection_result["processed_image_path"],
                    timestamp=datetime.utcnow()
                )
                detections_to_save_to_db.append(new_db_detection)

        # Salva todas as detecções no banco de dados
        db.add_all(detections_to_save_to_db)
        db.commit()
        for det in detections_to_save_to_db:  # Para ter certeza de que o ID do DB é preenchido
            db.refresh(det)

        # Gerar o nome do arquivo XLSX e salvar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_filename = f"detections_report_{timestamp}.xlsx"
        xlsx_file_path = save_detection_to_xlsx(all_flat_detections_for_xlsx, xlsx_filename)

        # Prepara o resumo para a resposta JSON (para o frontend)
        response_summary = []
        for det_data in all_flat_detections_for_xlsx:  # Usa os dados preparados para XLSX
            response_summary.append({
                "id": det_data["image_id"],
                "tipo": det_data["class"],
                "acuracia": f"{det_data['confidence'] * 100:.2f}%"
            })

        return JSONResponse(content={
            "status": "success",
            "message": "Imagens processadas e resultados salvos no DB e XLSX gerado.",
            "detections_summary": response_summary,
            "xlsx_report_url": f"/reports/{os.path.basename(xlsx_file_path)}"
        })


    except Exception as e:

        db.rollback()  # Em caso de erro, desfaz qualquer alteração no DB

        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")

    finally:

        for path in temp_image_paths:

            if os.path.exists(path):
                os.remove(path)

@router.get("/processed_images/{image_name}")
async def get_processed_image(image_name: str):
    """
    Endpoint para servir imagens processadas (com BBOXs desenhados).
    """
    image_path = PROCESSED_IMAGES_DIR / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Imagem processada não encontrada.")
    return FileResponse(image_path)


@router.get("/reports/{report_name}")
async def get_report_xlsx(report_name: str):
    """
    Endpoint para servir relatórios XLSX.
    """
    report_path = XLSX_RESULTS_DIR / report_name
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Relatório XLSX não encontrado.")
    return FileResponse(report_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")