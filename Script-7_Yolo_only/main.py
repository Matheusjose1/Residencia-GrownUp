from ultralytics import YOLO
import cv2
import numpy as np
import os

# Carrega modelo YOLO
model = YOLO("yolov8n.pt")

# Cria pasta de saída, se não existir
os.makedirs("output", exist_ok=True)

def detectar_objetos(imagem_path, nome_saida):
    resultados = model(imagem_path)
    objetos = []
    for resultado in resultados:
        for det in resultado.boxes:
            classe = int(det.cls)
            bbox = det.xyxy.cpu().numpy().flatten()
            objetos.append((classe, bbox))

        # Desenhar caixas na imagem
        imagem = cv2.imread(imagem_path)
        for det in resultado.boxes:
            x1, y1, x2, y2 = det.xyxy[0].int().tolist()
            cls_id = int(det.cls[0])
            label = model.names[cls_id]
            cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(imagem, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join("output", f"processed_{nome_saida}")
        cv2.imwrite(output_path, imagem)

    return objetos

def calcular_similaridade(objetos1, objetos2, iou_threshold=0.5):
    correspondencias = 0
    usados = set()

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    for i, (classe1, box1) in enumerate(objetos1):
        for j, (classe2, box2) in enumerate(objetos2):
            if j in usados:
                continue
            if classe1 == classe2 and iou(box1, box2) > iou_threshold:
                correspondencias += 1
                usados.add(j)
                break

    total_objetos = max(len(objetos1), len(objetos2))
    similaridade = (correspondencias / total_objetos) * 100 if total_objetos > 0 else 100
    return similaridade, correspondencias, len(objetos1), len(objetos2)

def comparar_imagens(img1_path, img2_path):
    obj1 = detectar_objetos(img1_path, os.path.basename(img1_path))
    obj2 = detectar_objetos(img2_path, os.path.basename(img2_path))

    sim, correspondencias, total1, total2 = calcular_similaridade(obj1, obj2)

    if sim == 100:
        resultado = "Verdadeiro Positivo (100% de similaridade)"
    elif sim >= 70:
        resultado = "Parcialmente semelhantes (alto grau de similaridade)"
    elif 30 < sim < 70:
        resultado = "Falso Positivo/Negativo (semelhança média)"
    else:
        resultado = "Verdadeiro Negativo (baixa similaridade)"

    print(f"📷 Comparação entre imagens:")
    print(f"- Objetos na imagem 1: {total1}")
    print(f"- Objetos na imagem 2: {total2}")
    print(f"- Correspondências: {correspondencias}")
    print(f"- Similaridade: {sim:.2f}%")
    print(f"🔍 Resultado: {resultado}")

# Exemplo de uso
img1 = "Imagens/PAP-0026 Cad.jpg"
img2 = "Imagens/PAP-0026 Op.jpg"
comparar_imagens(img1, img2)
