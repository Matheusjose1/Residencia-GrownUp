# pip install ultralytics
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os

# Caminho da pasta com imagens
image_folder = 'caminho/para/sua/pasta_de_imagens'  # <-- Altere aqui
output_folder = 'output_imagens'  # Pasta onde as imagens com resultado serão salvas

# Cria a pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Carrega o modelo YOLOv8
model = YOLO('yolov8n.pt')

# Define a(s) classe(s) que você quer contar (ex: 2 = carro)
classes_to_count = [2]

# Região onde os objetos serão contados
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Inicializa o contador de objetos
counter = object_counter.ObjectCounter()
counter.set_args(
    view_img=False,  # Define como True se quiser ver na tela também
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True
)

# Processa todas as imagens da pasta
for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(image_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        # Carrega a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f'Erro ao carregar a imagem: {img_name}')
            continue

        # Aplica o modelo de detecção e rastreamento
        tracks = model.track(image, persist=True, show=False, classes=classes_to_count)

        # Aplica a contagem na imagem
        result_img = counter.start_counting(image, tracks)

        # Salva a imagem processada
        cv2.imwrite(output_path, result_img)
        print(f'Imagem processada e salva: {output_path}')
