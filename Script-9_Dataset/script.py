from ultralytics import YOLO
import cv2

# Carrega o modelo treinado
model = YOLO('runs/detect/lixeiras-yolo8/weights/best.pt')

def detectar_lixeira(imagem_path):
    results = model(imagem_path)
    classes = results[0].boxes.cls.cpu().numpy()
    nomes = [model.names[int(c)] for c in classes]
    return nomes

def comparar_lixeiras(img1_path, img2_path):
    classes1 = detectar_lixeira(img1_path)
    classes2 = detectar_lixeira(img2_path)

    print(f"Imagem 1: {classes1}")
    print(f"Imagem 2: {classes2}")

    return classes1 == classes2

# Teste
img1 = 'Imagens/CAP-0005.jpeg'
img2 = 'exemplos/lixeira2.jpg'

if comparar_lixeiras(img1, img2):
    print("IGUAIS.")
else:
    print("DIFERENTES.")
