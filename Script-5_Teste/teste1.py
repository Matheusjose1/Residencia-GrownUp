import cv2
import numpy as np

def compare_images_orb(img1_path, img2_path, threshold=70):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False, 0.0, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    similarity = len(matches) / max(len(kp1), len(kp2)) * 100
    is_same = similarity >= threshold

    return is_same, similarity, matches, kp1, kp2, img1, img2

def compare_images_sift(img1_path, img2_path, threshold=70):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False, 0.0, []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    similarity = len(good) / max(len(kp1), len(kp2)) * 100
    is_same = similarity >= threshold

    return is_same, similarity, good, kp1, kp2, img1, img2


def show_matches(img1, img2, kp1, kp2, matches, title="Matches"):
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
    cv2.imshow(title, matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

is_same, similarity, matches, kp1, kp2, img1, img2 = compare_images_orb("Imagens/Lixeira-Perfil.png", "Imagens/ExemploLixeira.png")
print(f"[ORB] Similaridade: {similarity:.2f}%")
print("Mesma lixeira? ", "SIM" if is_same else "NÃO")
show_matches(img1, img2, kp1, kp2, matches)

is_same, similarity, good_matches, kp1, kp2, img1, img2 = compare_images_sift("Imagens/Lixeira-Perfil.png", "Imagens/ExemploLixeira.png")
print(f"[SIFT] Similaridade: {similarity:.2f}%")
print("Mesma lixeira? ", "SIM" if is_same else "NÃO")
show_matches(img1, img2, kp1, kp2, [m[0] for m in good_matches], title="SIFT Matches")
