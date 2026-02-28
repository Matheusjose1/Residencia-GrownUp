# app/core/config.py
import os
from pathlib import Path
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

# IMPORTS: Todos no topo do arquivo.
from torch.nn import Sequential, Conv2d, BatchNorm2d, SiLU
try:
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF, C3k2
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import DetectionModel
except ImportError as e:
    print(f"ATENÇÃO: Não foi possível importar um ou mais módulos YOLO necessários para add_safe_globals: {e}")
    print("Isso pode acontecer se a estrutura interna do Ultralytics mudar. O carregamento pode falhar.")
    # Defina-os como None ou pule para evitar que a linha add_safe_globals quebre.
    # Comente os que não puder importar para testar.
    Conv = None
    C2f = None
    C3k2 = None
    Bottleneck = None
    SPPF = None
    Detect = None
    DetectionModel = None


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
torch.serialization.add_safe_globals([DetectionModel])
YOLO_MODEL_PATH = BASE_DIR / "training" / "yolov11_lixeiras_custom" / "weights" / "best.pt"
YOLO_GENERAL_MODEL_PATH = None

# Carregamento dos modelos
try:

    safe_globals_list = [
        # Módulos PyTorch genéricos que podem ser serializados pelo modelo
        Sequential,
        Conv2d,
        BatchNorm2d,
        SiLU,
        C3k2,
        Conv,
        C2f,
        Bottleneck,
        SPPF,
        Detect,
        DetectionModel
    ]
    # Filtra None para o caso de algum import ter falhado no try/except acima
    torch.serialization.add_safe_globals([g for g in safe_globals_list if g is not None])


    if YOLO_MODEL_PATH.exists():
        model_yolo_lixeiras = YOLO(YOLO_MODEL_PATH)
        print(f"Modelo YOLO de lixeiras carregado com sucesso de: {YOLO_MODEL_PATH}")
    else:
        print(f"ERRO: Modelo YOLO não encontrado em: {YOLO_MODEL_PATH}")
        model_yolo_lixeiras = None

except Exception as e:
    print(f"Erro ao carregar modelo YOLO de lixeiras: {e}")
    model_yolo_lixeiras = None

# Classes esperadas do seu modelo YOLO
YOLO_CLASSES = {
    0: "domiciliar",
    1: "volumoso",
    2: "poda",
}