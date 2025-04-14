import cv2
import numpy as np

# Carrega imagem de referencia
reference_image = cv2.imread('Imagens/Lixeira-Perfil.png')

# Converte imagem para tons de cinza
gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Inicializa Sift
sift = cv2.SIFT_create()

#Detecta pontos chave e computa descritores para a imagem de referência
keypoints_reference, descriptors_reference = sift.detectAndCompute(gray_reference, None)

#Converte descritores para o tipo CV_32f
descriptors_reference = descriptors_reference.astype(np.float32)

#Inicializa a camêra(webcan)

# Usuário deve ser capaz de identificar imagens pela sua camêra?

cap = cv2.VideoCapture(0)

# Seleciona  a área para ser detectada
bbox = cv2.selectROI('Select ROI', reference_image, fromCenter=False, showCrosshair=True)
x, y, w, h = bbox

# Extrai pontos chave e descritores da região selecionada

roi = gray_reference[y:y + h, x:x + w]
keypoints_roi, descriptors_roi = sift.detectAndCompute(roi, None)

# Converte descritores da area selecionada para o tipo CV_32F

descriptors_roi = descriptors_roi.astype(np.float32)

# Inicializa BFMatcher
bf = cv2.BFMatcher()

while True:
    # Captura video frame por frame
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecta pontos chaves e computa descritores do frame
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

    #Converte novamente os descritores para CV_32
    descriptors_frame = descriptors_frame.astype(np.float32)

    #Da match com os descritores do frame e da imagem de referência
    matches = bf.knnMatch(descriptors_roi, descriptors_frame, k=2)

    # Testa para encontrar bons matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calcula homografia
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints_roi[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            # Desenha uma caixa no objeto identificado
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2)

    # Mostra o frame do objeto detectado
    cv2.imshow('Objeto detectado', frame)

    # Pressionar q para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Realiza a captura
cap.release()
cv2.destroyAllWindows()