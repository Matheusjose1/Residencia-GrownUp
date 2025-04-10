import cv2
import numpy as np

#Carregando imagens
image1 = cv2.imread('1.png')
image2 = cv2.imread('2.png')

#Convertendo para escala cinza

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#Inicializando SIFT

sift = cv2.SIFT_create()

#Detectando pontos chave e descritores

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

#Iniciando Brute Force matcher

bf = cv2.BFMatcher()

#Match descritores

matches = bf.knnMatch(descriptors1, descriptors2, k=2)

#Ratio teste para encontrar bons matches

good_matches = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

#Extraindo pontos chaves dos matches

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)

dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

#Encontra homografia

H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#Quebra imagem 1 para 2

stitched_image = cv2.warpPerspective(image1, H, (image2.shape[1] + image1.shape[1], image2.shape[0]))
stitched_image[:, :image2.shape[1]] = image2

#Mostra imagem stitched

cv2.imshow('Stitched Image', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()