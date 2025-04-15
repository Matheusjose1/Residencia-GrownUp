# pip install ultralytics shapely opencv-python
from ultralytics.solutions import object_counter
import cv2
import os

# Caminho da pasta com imagens
image_folder = 'Imagens'  # <-- Altere aqui
output_folder = 'output_imagens'

# Cria a pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Região onde os objetos serão contados
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Inicializa o contador de objetos com as configurações
counter = object_counter.ObjectCounter(
    view_img=True,
    reg_pts=region_points,
    classes_names=None,      # Deixe como None para pegar automaticamente
    draw_tracks=True,
    classes=[2]              # Aqui você define quais classes quer contar, ex: 2 = carro
)

# Processa todas as imagens da pasta
# Processa todas as imagens da pasta
for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(image_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f'Erro ao carregar a imagem: {img_name}')
            continue

        # 👇 Cria uma nova instância para cada imagem
        counter = object_counter.ObjectCounter(
            view_img=False,
            reg_pts=region_points,
            classes_names=None,
            draw_tracks=False,
            classes=[2]
        )

        result_img = counter(image)
        cv2.imwrite(output_path, result_img)
        print(f'Imagem processada e salva: {output_path}')
