import cv2
import numpy as np

original = cv2.imread('Exemplo2.png')
image_to_compare = cv2.imread('ExemploLixeira2.png')

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

good = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

final_image = cv2.drawMatchesKnn(original,kp_1,image_to_compare, kp_2, good, None)
final_image = cv2.resize(final_image,(1000,650))

print(len(matches))

cv2.imshow('matches', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()