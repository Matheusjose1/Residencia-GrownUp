import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import csv
import math
from datetime import datetime
from ultralytics import YOLO
import uuid

# Configurações
THRESHOLD = 70 #Limite mínimo para verdadeiro positivo
PESO_LIXEIRA = 0.6 # Métrica da identificação da lixeira
PESO_CONTEXTO = 0.4 # Métrica da identificação do contexto da lixeira

# Modelos YOLO
model_yolo = YOLO("C:/Users/mathe/Documents/Estudos/Residencia-GrownUp/src/treinamentos/yolo_lixeiras7/weights/best.pt") #Modelo específico para lixeiras
modelo_geral = YOLO("yolov8n.pt") #Modelo pré treinado da YOLO


def escolher_arquivo(label): # Escolhe imagem para ser comparada
    filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png")])
    if filepath:
        label.config(text=os.path.basename(filepath))
    return filepath

def salvar_csv(resultado_dict): # Gera tabela csv
    pasta_csv = os.path.join(pasta_base, "resultados csv")
    os.makedirs(pasta_csv, exist_ok=True)

    data_atual = datetime.now().strftime("%Y-%m-%d")
    nome_arquivo = f"resultado_comparacao_{data_atual}.csv"
    caminho_arquivo = os.path.join(pasta_csv, nome_arquivo)

    # Define a ordem fixa dos campos (inclui o campo 'id')
    fieldnames = [
        "id",
        "imagem_1",
        "imagem_2",
        "algoritmo",
        "similaridade_%", 
        "classificacao",
        "contexto_img1",
        "contexto_img2",
        "resultado_path"
    ]

    escrever_cabecalho = not os.path.exists(caminho_arquivo)
    with open(caminho_arquivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if escrever_cabecalho:
            writer.writeheader()
        writer.writerow(resultado_dict)

script_dir = os.path.dirname(os.path.abspath(__file__))
pasta_raiz = os.path.abspath(os.path.join(script_dir, os.pardir))
pasta_base =  os.path.join(pasta_raiz, "output")
pasta_imagens = os.path.join(pasta_base, 'imagens processadas')
pasta_csv = os.path.join(pasta_base, "resultados csv")

os.makedirs(pasta_imagens, exist_ok=True)
os.makedirs(pasta_csv, exist_ok=True)

def detectar_objetos_yolo(imagem_path, nome_saida):


    resultados = model_yolo(imagem_path)
    objetos = []
    for resultado in resultados:
        imagem = cv2.imread(imagem_path)
        for det in resultado.boxes:
            cls_id = int(det.cls[0])
            bbox = det.xyxy[0].int().tolist()
            objetos.append((cls_id, bbox))
            x1, y1, x2, y2 = bbox
            label = model_yolo.names[cls_id]
            cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(imagem, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(pasta_imagens, f"processed_{nome_saida}")
        cv2.imwrite(output_path, imagem)

        csv_path = os.path.join(pasta_csv, f'resultados_{os.path.splitext(nome_saida)[0]}.csv')
    return objetos, output_path

def detectar_contexto_geral(imagem_path):
    resultados = modelo_geral(imagem_path)
    contexto = []
    for resultado in resultados:
        for det in resultado.boxes:
            cls_id = int(det.cls[0])
            label = modelo_geral.names[cls_id]
            contexto.append(label)
    return contexto

def calcular_similaridade_yolo(objetos1, objetos2, dist_threshold=100):
    correspondencias = 0
    usados = set()
    def centro(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    for i, (cls1, bbox1) in enumerate(objetos1):
        c1 = centro(bbox1)
        for j, (cls2, bbox2) in enumerate(objetos2):
            if j in usados:
                continue
            c2 = centro(bbox2)
            if cls1 == cls2:
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                if dist < dist_threshold:
                    correspondencias += 1
                    usados.add(j)
                    break
    total = max(len(objetos1), len(objetos2))
    return (correspondencias / total) * 100 if total > 0 else 100

def comparar_sift_orb(img1_path, img2_path, algoritmo):
    if algoritmo == 'ORB':
        detector = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        detector = cv2.SIFT_create(nfeatures=2000)
        matcher = cv2.BFMatcher()
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return None, 0
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        matches_mask = mask.ravel().tolist() if mask is not None else None
        consistentes = sum(matches_mask) if matches_mask else 0
        similaridade = (consistentes / ((len(kp1) + len(kp2)) / 2)) * 100
    else:
        similaridade = 0
        matches_mask = None
    img_out = cv2.drawMatches(cv2.imread(img1_path), kp1, cv2.imread(img2_path), kp2, good[:50], None,
                              matchColor=(0, 255, 0),
                              matchesMask=matches_mask[:50] if matches_mask else None,
                              flags=2)
    return img_out, similaridade

def iniciar_comparacao():
    img1_path = file_1.get()
    img2_path = file_2.get()
    algoritmo = algoritmo_selecionado.get()

    if not img1_path or not img2_path:
        messagebox.showwarning("Erro", "Selecione duas imagens.")
        return

    id_comparacao = str(uuid.uuid4())  # Gera ID único para esta execução

    if algoritmo == "YOLO":
        objetos1, path1 = detectar_objetos_yolo(img1_path, "1.jpg")
        objetos2, path2 = detectar_objetos_yolo(img2_path, "2.jpg")
        similaridade_lixeiras = calcular_similaridade_yolo(objetos1, objetos2)

        contexto1 = detectar_contexto_geral(img1_path)
        contexto2 = detectar_contexto_geral(img2_path)
        intersecao = set(contexto1).intersection(set(contexto2))
        union = set(contexto1).union(set(contexto2))
        similaridade_contexto = (len(intersecao) / len(union)) * 100 if union else 100

        similaridade = (similaridade_lixeiras * PESO_LIXEIRA) + (similaridade_contexto * PESO_CONTEXTO)
        classificacao = "Verdadeiro Positivo (TP)" if similaridade >= THRESHOLD else "Verdadeiro Negativo (TN)"

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        altura = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * altura / img1.shape[0]), altura))
        img2 = cv2.resize(img2, (int(img2.shape[1] * altura / img2.shape[0]), altura))
        resultado = cv2.hconcat([img1, img2])
    else:
        resultado, similaridade = comparar_sift_orb(img1_path, img2_path, algoritmo)
        if resultado is None:
            messagebox.showerror("Erro", "Não foi possível detectar características.")
            return
        contexto1 = detectar_contexto_geral(img1_path)
        contexto2 = detectar_contexto_geral(img2_path)
        intersecao = set(contexto1).intersection(set(contexto2))
        union = set(contexto1).union(set(contexto2))
        similaridade_contexto = (len(intersecao) / len(union)) * 100 if union else 100
        similaridade_lixeiras = similaridade
        similaridade = (similaridade_lixeiras * PESO_LIXEIRA) + (similaridade_contexto * PESO_CONTEXTO)
        classificacao = "Verdadeiro Positivo (TP)" if similaridade >= THRESHOLD else "Verdadeiro Negativo (TN)"

    texto_resultado.set(f"Similaridade Total: {similaridade:.2f}%\nResultado: {classificacao}")
    barra_lixeira['value'] = similaridade_lixeiras
    barra_contexto['value'] = similaridade_contexto

    resultado_filename = f"resultado_{os.path.basename(img1_path)}_{os.path.basename(img2_path)}.jpg"
    resultado_path = os.path.join(pasta_imagens, resultado_filename)
    cv2.imwrite(resultado_path, resultado)
    img = Image.open(resultado_path)
    img.thumbnail((850, 500))
    img_tk = ImageTk.PhotoImage(img)
    imagem_resultado.config(image=img_tk)
    imagem_resultado.image = img_tk

    salvar_csv({
        "id": id_comparacao,
        "imagem_1": os.path.basename(img1_path),
        "imagem_2": os.path.basename(img2_path),
        "algoritmo": algoritmo,
        "similaridade_%": round(similaridade, 2),
        "classificacao": classificacao,
        "contexto_img1": ", ".join(contexto1),
        "contexto_img2": ", ".join(contexto2),
        "resultado_path": resultado_path
    })

# Interface gráfica
root = tk.Tk()
root.title("Comparador de Lixeiras")
root.geometry("950x750")

frame = tk.Frame(root)
frame.pack(pady=20)

file_1 = tk.StringVar()
file_2 = tk.StringVar()
algoritmo_selecionado = tk.StringVar(value="SIFT")
texto_resultado = tk.StringVar()

tk.Button(frame, text="Selecionar Imagem 1", command=lambda: file_1.set(escolher_arquivo(label_img1))).grid(row=0, column=0, padx=5, pady=5)
label_img1 = tk.Label(frame, text="Nenhuma imagem")
label_img1.grid(row=0, column=1, padx=5)

tk.Button(frame, text="Selecionar Imagem 2", command=lambda: file_2.set(escolher_arquivo(label_img2))).grid(row=1, column=0, padx=5, pady=5)
label_img2 = tk.Label(frame, text="Nenhuma imagem")
label_img2.grid(row=1, column=1, padx=5)

tk.Label(frame, text="Algoritmo:").grid(row=2, column=0, pady=10)
ttk.Combobox(frame, textvariable=algoritmo_selecionado, values=["SIFT", "ORB", "YOLO"], state="readonly").grid(row=2, column=1)

tk.Button(frame, text="Comparar Imagens", command=iniciar_comparacao, bg="green", fg="white").grid(row=3, columnspan=2, pady=10)

tk.Label(root, textvariable=texto_resultado, font=("Arial", 14)).pack(pady=10)

tk.Label(root, text="Similaridade de Lixeiras").pack()
barra_lixeira = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate", maximum=100)
barra_lixeira.pack(pady=5)

tk.Label(root, text="Similaridade de Contexto").pack()
barra_contexto = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate", maximum=100)
barra_contexto.pack(pady=5)

imagem_resultado = tk.Label(root)
imagem_resultado.pack()

root.mainloop()