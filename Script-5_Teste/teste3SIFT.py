import cv2
import numpy as np
import os
import csv
from itertools import combinations

THRESHOLD = 70  # % de similaridade mínima

def comparar_orb(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or not kp1 or not kp2:
        return False, 0.0, [], kp1, kp2, img1, img2

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    similarity = len(matches) / max(len(kp1), len(kp2)) * 100
    return similarity >= THRESHOLD, similarity, matches, kp1, kp2, img1, img2

def comparar_sift(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or not kp1 or not kp2:
        return False, 0.0, [], kp1, kp2, img1, img2

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    similarity = len(good) / max(len(kp1), len(kp2)) * 100
    return similarity >= THRESHOLD, similarity, good, kp1, kp2, img1, img2

def desenhar_matches(img1, img2, kp1, kp2, matches, output_path):
    resultado = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
    cv2.imwrite(output_path, resultado)

def processar_imagens(diretorio_imagens, algoritmo='orb', salvar_imgs=True):
    imagens = [os.path.join(diretorio_imagens, f) for f in os.listdir(diretorio_imagens) if f.lower().endswith(('.jpg', '.png'))]
    resultados = []
    os.makedirs("matches", exist_ok=True)

    for img1_path, img2_path in combinations(imagens, 2):
        if algoritmo == 'orb':
            mesma, sim, matches, kp1, kp2, img1, img2 = comparar_orb(img1_path, img2_path)
        elif algoritmo == 'sift':
            mesma, sim, matches, kp1, kp2, img1, img2 = comparar_sift(img1_path, img2_path)
        else:
            raise ValueError("Algoritmo inválido. Use 'orb' ou 'sift'.")

        nome1 = os.path.basename(img1_path)
        nome2 = os.path.basename(img2_path)
        match_img_path = f"matches/{nome1.replace('.jpg', '')}_{nome2.replace('.jpg', '')}_{algoritmo}.jpg"

        if salvar_imgs:
            desenhar_matches(img1, img2, kp1, kp2, matches, match_img_path)

        resultados.append({
            'imagem_1': nome1,
            'imagem_2': nome2,
            'algoritmo': algoritmo.upper(),
            'similaridade_%': round(sim, 2),
            'mesma_lixeira': 'SIM' if mesma else 'NÃO',
            'imagem_matches': match_img_path
        })

        print(f"{nome1} vs {nome2} → {sim:.2f}% → {'SIM' if mesma else 'NÃO'}")

    return resultados

def salvar_csv(resultados, arquivo_csv="resultados.csv"):
    campos = ['imagem_1', 'imagem_2', 'algoritmo', 'similaridade_%', 'mesma_lixeira', 'imagem_matches']
    with open(arquivo_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)
    print(f"\n✅ CSV salvo como: {arquivo_csv}")

if __name__ == "__main__":
    pasta_imagens = "imagens"
    algoritmo = 'sift'  # escolha: 'orb' ou 'sift'

    print(f"\n🔎 Iniciando comparações com algoritmo: {algoritmo.upper()}")
    resultados = processar_imagens(pasta_imagens, algoritmo=algoritmo)
    salvar_csv(resultados)
