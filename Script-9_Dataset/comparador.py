import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import csv
import math
from ultralytics import YOLO

# Porcentagem mínima para verdadeiro positivo
THRESHOLD = 70

# Carregar o modelo YOLO treinado
model_yolo = YOLO("treinamentos/yolo_lixeiras7/weights/best.pt")

# Criar pasta de saída se não existir
os.makedirs("output", exist_ok=True)

# === FUNÇÕES ===

def escolher_arquivo(label):
    filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png")])
    if filepath:
        label.config(text=os.path.basename(filepath))
    return filepath

def salvar_csv(resultado_dict):
    arquivo_csv = "resultado_comparacao.csv"
    escrever_cabecalho = not os.path.exists(arquivo_csv)
    with open(arquivo_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=resultado_dict.keys())
        if escrever_cabecalho:
            writer.writeheader()
        writer.writerow(resultado_dict)

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

        output_path = os.path.join("output", f"processed_{nome_saida}")
        cv2.imwrite(output_path, imagem)

    return objetos, output_path

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

    if algoritmo == "YOLO":
        objetos1, path1 = detectar_objetos_yolo(img1_path, "1.jpg")
        objetos2, path2 = detectar_objetos_yolo(img2_path, "2.jpg")
        similaridade = calcular_similaridade_yolo(objetos1, objetos2)

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

    # Classificação
    mesma_pred = similaridade >= THRESHOLD
    mesma_real = rotulo_real.get()

    if mesma_pred and mesma_real:
        classificacao = "Verdadeiro Positivo (TP)"
    elif mesma_pred and not mesma_real:
        classificacao = "Falso Positivo (FP)"
    elif not mesma_pred and not mesma_real:
        classificacao = "Verdadeiro Negativo (TN)"
    else:
        classificacao = "Falso Negativo (FN)"

    texto_resultado.set(f"Similaridade: {similaridade:.2f}%\nResultado: {classificacao}")

    resultado_path = f"resultado_{os.path.basename(img1_path)}_{os.path.basename(img2_path)}.jpg"
    cv2.imwrite(resultado_path, resultado)

    img = Image.open(resultado_path)
    img.thumbnail((850, 500))
    img_tk = ImageTk.PhotoImage(img)
    imagem_resultado.config(image=img_tk)
    imagem_resultado.image = img_tk

    salvar_csv({
        "imagem_1": os.path.basename(img1_path),
        "imagem_2": os.path.basename(img2_path),
        "algoritmo": algoritmo,
        "similaridade_%": round(similaridade, 2),
        "classificacao": classificacao,
        "resultado_path": resultado_path
    })

# === INTERFACE ===

root = tk.Tk()
root.title("Comparador de Lixeiras")
root.geometry("950x700")

frame = tk.Frame(root)
frame.pack(pady=20)

file_1 = tk.StringVar()
file_2 = tk.StringVar()
algoritmo_selecionado = tk.StringVar(value="SIFT")
texto_resultado = tk.StringVar()
rotulo_real = tk.BooleanVar(value=False)

# Inputs
tk.Button(frame, text="Selecionar Imagem 1", command=lambda: file_1.set(escolher_arquivo(label_img1))).grid(row=0, column=0, padx=5, pady=5)
label_img1 = tk.Label(frame, text="Nenhuma imagem")
label_img1.grid(row=0, column=1, padx=5)

tk.Button(frame, text="Selecionar Imagem 2", command=lambda: file_2.set(escolher_arquivo(label_img2))).grid(row=1, column=0, padx=5, pady=5)
label_img2 = tk.Label(frame, text="Nenhuma imagem")
label_img2.grid(row=1, column=1, padx=5)

tk.Label(frame, text="Algoritmo:").grid(row=2, column=0, pady=10)
ttk.Combobox(frame, textvariable=algoritmo_selecionado, values=["SIFT", "ORB", "YOLO"], state="readonly").grid(row=2, column=1)

tk.Button(frame, text="Comparar Imagens", command=iniciar_comparacao, bg="green", fg="white").grid(row=4, columnspan=2, pady=10)

tk.Label(root, textvariable=texto_resultado, font=("Arial", 14)).pack(pady=10)
imagem_resultado = tk.Label(root)
imagem_resultado.pack()

root.mainloop()
