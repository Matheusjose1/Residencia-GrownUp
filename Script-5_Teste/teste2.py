import cv2
import numpy as np
import os
import csv
from itertools import combinations

THRESHOLD = 70  # % de similaridade para considerar como a mesma lixeira

def comparar_imagens_orb(img1_path, img2_path, threshold=THRESHOLD):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return False, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    similarity = len(matches) / max(len(kp1), len(kp2)) * 100
    return similarity >= threshold, similarity

def processar_imagens(diretorio_imagens, algoritmo='orb'):
    imagens = [os.path.join(diretorio_imagens, f) for f in os.listdir(diretorio_imagens) if f.endswith(('.jpg', '.png'))]
    resultados = []

    print(f"🔍 Comparando {len(imagens)} imagens...")

    for img1_path, img2_path in combinations(imagens, 2):
        if algoritmo == 'orb':
            mesma_lixeira, similaridade = comparar_imagens_orb(img1_path, img2_path)
        else:
            raise NotImplementedError("Algoritmo não suportado")

        resultado = {
            'imagem_1': os.path.basename(img1_path),
            'imagem_2': os.path.basename(img2_path),
            'similaridade_%': round(similaridade, 2),
            'mesma_lixeira': 'SIM' if mesma_lixeira else 'NÃO'
        }

        resultados.append(resultado)
        print(f"{resultado['imagem_1']} vs {resultado['imagem_2']} → {resultado['similaridade_%']}% → {resultado['mesma_lixeira']}")

    return resultados

def salvar_csv(resultados, caminho_csv='resultados.csv'):
    with open(caminho_csv, mode='w', newline='', encoding='utf-8') as arquivo:
        campos = ['imagem_1', 'imagem_2', 'similaridade_%', 'mesma_lixeira']
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n📁 Resultados salvos em: {caminho_csv}")

if __name__ == "__main__":
    pasta_imagens = "imagens"  # diretório com imagens
    resultados = processar_imagens(pasta_imagens)
    salvar_csv(resultados)
