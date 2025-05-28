import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from datetime import datetime

from app.models.image_processing import process_single_image_yolo
from app.core.utils import save_detection_to_xlsx, extract_id_from_filename
from app.core.config import PROCESSED_IMAGES_DIR, XLSX_RESULTS_DIR

router = APIRouter()


@router.post("/process_images/")
async def process_images(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    Endpoint para processar uma ou múltiplas imagens usando o modelo YOLO.
    Recebe um ou mais arquivos de imagem e retorna um JSON com o resumo das detecções
    e um link para o arquivo XLSX de resultados, contendo ID, Tipo e Acurácia.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    temp_image_paths = []
    # Lista para coletar TODOS os itens de detecção individuais para o XLSX
    all_flat_detections = []

    try:
        for image_file in files:
            if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise HTTPException(status_code=400,
                                    detail=f"Apenas arquivos JPG ou PNG são permitidos. Recebido: {image_file.filename}")

            # Salva a imagem temporariamente
            temp_path = os.path.join("/tmp", image_file.filename)
            os.makedirs("/tmp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(await image_file.read())
            temp_image_paths.append(temp_path)

            # Processa a imagem com YOLO
            detection_result = process_single_image_yolo(temp_path)

            # Adiciona as detecções desta imagem à lista global para o XLSX
            # Note que 'detection_result["detections"]' já está no formato {"image_id", "class", "confidence"}
            all_flat_detections.extend(detection_result["detections"])

            # Se não houver detecções, adiciona uma linha indicando isso no XLSX
            if not detection_result["detections"]:
                image_id = extract_id_from_filename(detection_result["image_name"]) or "N/A"
                all_flat_detections.append({
                    "image_id": image_id,
                    "class": "Nenhuma detecção",
                    "confidence": 0.0
                })

        # Gerar o nome do arquivo XLSX baseado na data/hora
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_filename = f"detections_report_{timestamp}.xlsx"

        # Salva todos os dados de detecção no XLSX
        xlsx_file_path = save_detection_to_xlsx(all_flat_detections, xlsx_filename)

        # Prepara o resumo para a resposta JSON (para o frontend)
        response_summary = []
        for det_data in all_flat_detections:
            # Para o resumo, vamos agrupar por imagem novamente se necessário
            # Ou retornar uma lista mais simples se a granularidade por detecção for ok
            # Para manter um resumo similar ao anterior:
            image_name = det_data["image_id"]  # Usando o ID como nome para o resumo

            # Para evitar duplicatas no resumo se uma imagem tiver múltiplas detecções
            # Isso é mais complexo, mas para um resumo simples, podemos listar todas as detecções
            response_summary.append({
                "id": det_data["image_id"],
                "tipo": det_data["class"],
                "acuracia": f"{det_data['confidence'] * 100:.2f}%"
            })

        return JSONResponse(content={
            "status": "success",
            "message": "Imagens processadas com sucesso. Relatório XLSX gerado.",
            "detections_summary": response_summary,  # Agora reflete a estrutura do XLSX
            "xlsx_report_url": f"/reports/{os.path.basename(xlsx_file_path)}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")
    finally:
        # Limpa as imagens temporárias
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