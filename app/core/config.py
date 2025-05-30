import os
from pathlib import Path
from ultralytics import YOLO
import torch


# Caminhos base
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Configurações de diretórios
OUTPUT_DIR = BASE_DIR / "data" / "output"
PROCESSED_IMAGES_DIR = OUTPUT_DIR / "imagens_processadas"
XLSX_RESULTS_DIR = OUTPUT_DIR / "resultados_xlsx"

# Garante que os diretórios existam
PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
XLSX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Caminhos dos modelos YOLO
YOLO_LIXEIRAS_MODEL_PATH = BASE_DIR / "training" / "yolov11_residuos_custom2" / "weights" / "best.pt"
YOLO_GENERAL_MODEL_PATH = None

# Carregamento dos modelos
try:
    model_yolo_lixeiras = YOLO(YOLO_LIXEIRAS_MODEL_PATH)
    print(f"Modelo YOLO de lixeiras carregado com sucesso de: {YOLO_LIXEIRAS_MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar modelo YOLO de lixeiras: {e}")
    model_yolo_lixeiras = None

# Classes esperadas do seu modelo YOLO
YOLO_CLASSES = {
    0: "domiciliar",
    1: "volumoso",
    2: "poda",
}