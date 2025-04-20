from ultralytics import YOLO

# Carregar modelo pré-treinado
model = YOLO('yolov8s.pt')  # ou yolov8n.pt para mais leve

# Treinar com o dataset
model.train(
    data='lixeiras.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='lixeiras-yolo8',
)
