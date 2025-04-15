import tkinter as tk
from tkinter import filedialog
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

    return objetos, f"output/processed_{nome_saida}"

# Função de similaridade (IoU)
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
# GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparador de Imagens com YOLO")
        self.root.geometry("1000x600")

        self.img1_path = None
        self.img2_path = None

        self.label_info = tk.Label(root, text="Selecione duas imagens para comparar", font=("Arial", 14))
        self.label_info.pack(pady=10)

        self.btn1 = tk.Button(root, text="Selecionar Imagem 1", command=self.load_img1)
        self.btn1.pack()

        self.btn2 = tk.Button(root, text="Selecionar Imagem 2", command=self.load_img2)
        self.btn2.pack()

        self.compare_btn = tk.Button(root, text="Comparar Imagens", command=self.comparar, state=tk.DISABLED)
        self.compare_btn.pack(pady=10)

        self.canvas = tk.Canvas(root, width=960, height=360)
        self.canvas.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def load_img1(self):
        path = filedialog.askopenfilename()
        if path:
            self.img1_path = path
            self.check_ready()

    def load_img2(self):
        path = filedialog.askopenfilename()
        if path:
            self.img2_path = path
            self.check_ready()

    def check_ready(self):
        if self.img1_path and self.img2_path:
            self.compare_btn.config(state=tk.NORMAL)

    def comparar(self):
        objetos1, out1 = detectar_objetos(self.img1_path, os.path.basename(self.img1_path))
        objetos2, out2 = detectar_objetos(self.img2_path, os.path.basename(self.img2_path))
        similaridade = calcular_similaridade(objetos1, objetos2)

        # Mostrar imagens
        img1 = Image.open(out1).resize((480, 360))
        img2 = Image.open(out2).resize((480, 360))
        img1_tk = ImageTk.PhotoImage(img1)
        img2_tk = ImageTk.PhotoImage(img2)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img1_tk)
        self.canvas.create_image(480, 0, anchor=tk.NW, image=img2_tk)

        self.canvas.image1 = img1_tk
        self.canvas.image2 = img2_tk

        self.result_label.config(text=f"Similaridade: {similaridade:.2f}%")

# Rodar app
root = tk.Tk()
app = App(root)
root.mainloop()
