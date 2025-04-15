import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

THRESHOLD = 70

def comparar_imagens(img1_path, img2_path, algoritmo):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    if algoritmo == 'ORB':
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif algoritmo == 'SIFT':
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher()
    else:
        raise ValueError("Algoritmo inválido")

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None, 0, None

    if algoritmo == 'SIFT':
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    else:
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        consistentes = sum(matches_mask)
        similaridade = consistentes / max(len(kp1), len(kp2)) * 100
    else:
        similaridade = 0
        matches_mask = None

    resultado = cv2.drawMatches(
        cv2.imread(img1_path), kp1,
        cv2.imread(img2_path), kp2,
        good_matches[:20], None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask[:20] if matches_mask else None,
        flags=2
    )

    return resultado, similaridade, good_matches


def escolher_arquivo(label):
    filepath = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png")])
    if filepath:
        label.config(text=os.path.basename(filepath))
    return filepath

def iniciar_comparacao():
    img1_path = file_1.get()
    img2_path = file_2.get()
    algoritmo = algoritmo_selecionado.get()

    if not img1_path or not img2_path:
        messagebox.showwarning("Erro", "Selecione duas imagens.")
        return

    resultado, similaridade, matches = comparar_imagens(img1_path, img2_path, algoritmo)

    if resultado is None:
        messagebox.showerror("Erro", "Não foi possível detectar características.")
        return

    texto_resultado.set(f"Similaridade: {similaridade:.2f}% → {'MESMA' if similaridade >= THRESHOLD else 'DIFERENTE'}")

    cv2.imwrite("resultado_match.jpg", resultado)
    img = Image.open("resultado_match.jpg")
    img.thumbnail((800, 400))
    img_tk = ImageTk.PhotoImage(img)

    imagem_resultado.config(image=img_tk)
    imagem_resultado.image = img_tk

# --- GUI Tkinter ---
root = tk.Tk()
root.title("Comparador de Lixeiras")
root.geometry("900x600")

frame = tk.Frame(root)
frame.pack(pady=20)

file_1 = tk.StringVar()
file_2 = tk.StringVar()
algoritmo_selecionado = tk.StringVar(value="SIFT")
texto_resultado = tk.StringVar()

# Escolha de imagens
tk.Button(frame, text="Selecionar Imagem 1", command=lambda: file_1.set(escolher_arquivo(label_img1))).grid(row=0, column=0, padx=5, pady=5)
label_img1 = tk.Label(frame, text="Nenhuma imagem")
label_img1.grid(row=0, column=1, padx=5)

tk.Button(frame, text="Selecionar Imagem 2", command=lambda: file_2.set(escolher_arquivo(label_img2))).grid(row=1, column=0, padx=5, pady=5)
label_img2 = tk.Label(frame, text="Nenhuma imagem")
label_img2.grid(row=1, column=1, padx=5)

# Escolha do algoritmo
tk.Label(frame, text="Algoritmo:").grid(row=2, column=0, pady=10)
ttk.Combobox(frame, textvariable=algoritmo_selecionado, values=["SIFT", "ORB"], state="readonly").grid(row=2, column=1)

# Botão de comparar
tk.Button(frame, text="Comparar Imagens", command=iniciar_comparacao, bg="green", fg="white").grid(row=3, columnspan=2, pady=10)

# Resultado
tk.Label(root, textvariable=texto_resultado, font=("Arial", 14)).pack(pady=10)
imagem_resultado = tk.Label(root)
imagem_resultado.pack()

root.mainloop()
