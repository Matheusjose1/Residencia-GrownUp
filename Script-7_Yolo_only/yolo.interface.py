import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from ultralytics import YOLO
import math

# Carregar modelo YOLO
model = YOLO("yolov8n.pt")
os.makedirs("output", exist_ok=True)

# Função de detecção
def detectar_objetos(imagem_path, nome_saida):
    resultados = model(imagem_path)
    objetos = []

    for resultado in resultados:
        imagem = cv2.imread(imagem_path)
        for det in resultado.boxes:
            cls_id = int(det.cls[0])
            bbox = det.xyxy[0].int().tolist()
            objetos.append((cls_id, bbox))

            # Desenhar bounding box
            x1, y1, x2, y2 = bbox
            label = model.names[cls_id]
            cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(imagem, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Salvar imagem processada
        output_path = os.path.join("output", f"processed_{nome_saida}")
        cv2.imwrite(output_path, imagem)

    return objetos, output_path

# Função escolher arquivo
def escolher_arquivo(label):
    filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png")])
    if filepath:
        label.config(text=os.path.basename(filepath))
    return filepath

# Função de similaridade
def calcular_similaridade(objetos1, objetos2, dist_threshold=100):
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

# Função principal de comparação
def comparar():
    img1_path = file_1.get()
    img2_path = file_2.get()
    if not img1_path or not img2_path:
        texto_resultado.set("Por favor, selecione ambas as imagens.")
        return

    objetos1, path1 = detectar_objetos(img1_path, "1.jpg")
    objetos2, path2 = detectar_objetos(img2_path, "2.jpg")

    similaridade = calcular_similaridade(objetos1, objetos2)
    texto_resultado.set(f"Similaridade: {similaridade:.2f}%")

    # Mostrar as duas imagens lado a lado
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # Redimensionar para mesma altura (opcional, evita distorção)
    altura = min(img1.shape[0], img2.shape[0])
    img1_resized = cv2.resize(img1, (int(img1.shape[1] * altura / img1.shape[0]), altura))
    img2_resized = cv2.resize(img2, (int(img2.shape[1] * altura / img2.shape[0]), altura))

    concatenada = cv2.hconcat([img1_resized, img2_resized])
    imagem_rgb = cv2.cvtColor(concatenada, cv2.COLOR_BGR2RGB)
    imagem_pil = Image.fromarray(imagem_rgb)
    imagem_pil.thumbnail((850, 500))
    imagem_tk = ImageTk.PhotoImage(imagem_pil)

    imagem_resultado.configure(image=imagem_tk)
    imagem_resultado.image = imagem_tk

# GUI
root = tk.Tk()
root.title("Comparador de Imagens com YOLO")
root.geometry("950x700")

frame = tk.Frame(root)
frame.pack(pady=20)

file_1 = tk.StringVar()
file_2 = tk.StringVar()
algoritmo_selecionado = tk.StringVar(value="YOLO")
texto_resultado = tk.StringVar()
rotulo_real = tk.BooleanVar(value=False)

# Escolha de imagens
label_img1 = tk.Label(frame, text="Nenhuma imagem")
label_img1.grid(row=0, column=1, padx=5)
tk.Button(frame, text="Selecionar imagem 1", command=lambda:
          file_1.set(escolher_arquivo(label_img1))).grid(row=0, column=0, padx=5, pady=5)

label_img2 = tk.Label(frame, text="Nenhuma imagem")
label_img2.grid(row=1, column=1, padx=5)
tk.Button(frame, text="Selecionar imagem 2", command=lambda:
          file_2.set(escolher_arquivo(label_img2))).grid(row=1, column=0, padx=5, pady=5)

# Escolha do algoritmo (placeholder para expansão futura)
tk.Label(frame, text="Algoritmo:").grid(row=2, column=0, pady=10)
ttk.Combobox(frame, textvariable=algoritmo_selecionado, values=["YOLO"], state="readonly").grid(row=2, column=1)

# Botão de comparar
tk.Button(frame, text="Comparar Imagens", command=comparar, bg="green", fg="white").grid(row=4, columnspan=2, pady=10)

# Resultado
tk.Label(root, textvariable=texto_resultado, font=("Arial", 14)).pack(pady=10)
imagem_resultado = tk.Label(root)
imagem_resultado.pack()

root.mainloop()
