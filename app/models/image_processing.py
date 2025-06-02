import cv2
import os
from typing import List, Dict, Any, Tuple

from app.core.config import model_yolo_lixeiras, PROCESSED_IMAGES_DIR, YOLO_CLASSES
from app.core.utils import extract_id_from_filename  # Importar a função para extrair o ID


def process_single_image_yolo(image_path: str) -> Dict[str, Any]:
    """
    Processa uma única imagem usando o modelo YOLO para detectar lixeiras (domiciliar, volumoso, poda).
    Retorna um dicionário com os detalhes da detecção e o caminho da imagem processada.
    """
    if model_yolo_lixeiras is None:
        raise RuntimeError("Modelo YOLO de lixeiras não carregado. Verifique o caminho em config.py.")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem em: {image_path}")

    results = model_yolo_lixeiras(image_path)

    # Lista para armazenar detalhes de cada detecção para esta imagem
    detections_for_image: List[Dict[str, Any]] = []

    # Extrair ID da imagem do nome do arquivo
    image_base_name = os.path.basename(image_path)
    image_id = extract_id_from_filename(image_base_name)
    if image_id is None:
        image_id = "UNKNOWN_ID"  # Define um ID padrão se não puder ser extraído

    # Desenhar BBOXs e coletar dados de detecção
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            # bbox = box.xyxy[0].int().tolist() # Bounding box não será incluído no XLSX final

            class_name = YOLO_CLASSES.get(cls_id, f"Classe_{cls_id}")

            # Adiciona a detecção à lista no formato simplificado para o XLSX
            detections_for_image.append({
                "image_id": image_id,
                "class": class_name,
                "confidence": confidence
            })

            # Desenha no BBOX na imagem (ainda útil para visualização)
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            label_text = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Salva a imagem processada (com os BBOXs)
    processed_filename = f"detected_{image_base_name}"
    output_path = PROCESSED_IMAGES_DIR / processed_filename
    cv2.imwrite(str(output_path), img)

    return {
        "image_name": image_base_name,
        "processed_image_path": str(output_path),
        "detections": detections_for_image  # Lista de detecções simplificadas
    }