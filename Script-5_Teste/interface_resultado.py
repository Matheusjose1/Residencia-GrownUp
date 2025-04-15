import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import csv

THRESHOLD = 70

def comparar(img1_path, img2_path, detector, matcher, filtro_lowe=0.75):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, [], [], [], None

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < filtro_lowe * n.distance]

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        matches_mask = mask.ravel().tolist() if mask is not None else None
        consistentes = sum(matches_mask) if matches_mask else 0
        similaridade = consistentes / max(len(kp1), len(kp2)) * 100
    else:
        similaridade = 0
        matches_mask = None

    return similaridade, good_matches, kp1, kp2, matches_mask

def gerar_resultado(img1_path, img2_path, detector, matcher, algoritmo):
    similaridade, good_matches, kp1, kp2, matches_mask = comparar(img1_path, img2_path, detector, matcher)

    resultado = cv2.drawMatches(
        cv2.imread(img1_path), kp1,
        cv2.imread(img2_path), kp2,
        good_matches[:50], None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask[:50] if matches_mask else None,
        flags=2
    )

    return resultado, similaridade, good_matches

def comparar_imagens(img1_path, img2_path, algoritmo):
    if algoritmo == 'ORB':
        detector = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif algoritmo == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=2000)
        matcher = cv2.BFMatcher()
    else:
        raise ValueError("Algoritmo inválido")

    return gerar_resultado(img1_path, img2_path, detector, matcher, algoritmo)

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

    # Classificação com base no threshold e no rótulo real
    mesma_lixeira_predita = similaridade >= THRESHOLD
    mesma_lixeira_real = rotulo_real.get()

    if mesma_lixeira_predita and mesma_lixeira_real:
        classificacao = "Verdadeiro Positivo (TP)"
    elif mesma_lixeira_predita and not mesma_lixeira_real:
        classificacao = "Falso Positivo (FP)"
    elif not mesma_lixeira_predita and not mesma_lixeira_real:
        classificacao = "Verdadeiro Negativo (TN)"
    else:
        classificacao = "Falso Negativo (FN)"

    texto_resultado.set(f"Similaridade: {similaridade:.2f}%\nResultado: {classificacao}")

    # Salvar imagem do resultado
    nome_1 = os.path.basename(img1_path)
    nome_2 = os.path.basename(img2_path)
    img_resultado_path = f"resultado_{nome_1}_{nome_2}.jpg"
    cv2.imwrite(img_resultado_path, resultado)

    # Atualizar interface com imagem
    img = Image.open(img_resultado_path)
    img.thumbnail((800, 400))
    img_tk = ImageTk.PhotoImage(img)

    imagem_resultado.config(image=img_tk)
    imagem_resultado.image = img_tk

    # Salvar no CSV
    salvar_csv({
        "imagem_1": nome_1,
        "imagem_2": nome_2,
        "algoritmo": algoritmo,
        "similaridade_%": round(similaridade, 2),
        "classificacao": classificacao,
        "resultado_path": img_resultado_path
    })

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
rotulo_real = tk.BooleanVar(value=False)

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

# Rótulo verdadeiro
tk.Checkbutton(frame, text="As imagens são da mesma lixeira?", variable=rotulo_real).grid(row=3, columnspan=2)

# Botão de comparar
tk.Button(frame, text="Comparar Imagens", command=iniciar_comparacao, bg="green", fg="white").grid(row=4, columnspan=2, pady=10)

# Resultado
tk.Label(root, textvariable=texto_resultado, font=("Arial", 14)).pack(pady=10)
imagem_resultado = tk.Label(root)
imagem_resultado.pack()

root.mainloop()
