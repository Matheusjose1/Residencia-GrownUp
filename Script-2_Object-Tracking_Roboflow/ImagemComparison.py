import cv2
import numpy as np

#Compara explicitamente as imagens
'''
Comparando lixeira de perfil com lixeiras

original = cv2.imread('Imagens/Lixeira-Perfil.png')
image_to_compare = cv2.imread('Imagens/Lixeiras.png')


Comparando lixeira de perfil com uma lixeira distante

original = cv2.imread('Imagens/Lixeira-Perfil.png')
image_to_compare = cv2.imread('Imagens/ExemploLixeira.png')

Comparando várias lixeiras com lixeira de exemplo

original = cv2.imread('Imagens/Lixeiras.png')
image_to_compare = cv2.imread('Imagens/ExemploLixeira.png')

Comparando a lixeira de perfil com lixeira lateral

original = cv2.imread('Imagens/Lixeira-Perfil.png')
image_to_compare = cv2.imread('Imagens/Lixeira-Lateral.jpeg')
'''

original = cv2.imread('Imagens/Lixeira-Perfil.png')
image_to_compare = cv2.imread('Imagens/ExemploLixeira.png')

original = cv2.resize(original,(1000,650))
image_to_compare = cv2.resize(image_to_compare,(1000,650))

if original.shape == image_to_compare.shape:
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print('Imagens iguais')
    else:
        print('Imagens diferentes')

orb = cv2.ORB_create()
kp_1, desc_1 = orb.detectAndCompute(original, None)
kp_2, desc_2 = orb.detectAndCompute(original, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(desc_1, desc_2, k=2)

# Lista com os melhores matches

good = []

# Calculo dos melhores matches pela distância
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

final_image = cv2.drawMatchesKnn(original,kp_1,image_to_compare, kp_2, good, None)
final_image = cv2.resize(final_image,(1000,650))

# Printa todas as correspondências dentre as imagens
print(len(matches))

cv2.imshow('matches', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()