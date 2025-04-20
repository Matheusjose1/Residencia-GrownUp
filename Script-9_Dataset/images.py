from ultralytics import YOLO

# Caminho para o modelo pré-treinado
model = YOLO("yolov8n.pt")

# Treinar
model.train(
    data="lixeiras.yaml",   # esse arquivo deve existir e estar correto
    epochs=50,
    imgsz=640,
    batch=8,
    name="yolo_lixeiras",
    project="treinamentos"
)
