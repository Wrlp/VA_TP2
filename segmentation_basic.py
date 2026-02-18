import cv2
import numpy as np
import glob 
import os

# TODO 
"""
Membre 2 — Segmentation 1 : Méthodes simples (seuillage + morphologie)
🎯 Objectif
Implémenter des méthodes classiques et simples.

📌 Tâches
Conversion en niveaux de gris
Otsu
Seuillage adaptatif
Morphologie :
Opening / Closing
Érosion / Dilatation
Suppression des petits objets
Visualisation des grains détectés (contours)

📊 Dans le rapport
Quand ça marche ?
Quand ça échoue ?
Sensibilité au prétraitement ?
"""
# path input : results/preprocessing/
# path output : results/seg_basic/


def load_and_resize(path, size=(600, 600)):
    images = {}
    for file in glob.glob(os.path.join(path, "*.png")):
        img = cv2.imread(file)
        img = cv2.resize(img, size)
        name = os.path.basename(file)  
        images[name] = img

    return images

def show_changes(title, original, processed):
    combined = np.hstack((original, processed))
    cv2.imshow(title, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segmentation_basic(path):
    # TODO
    images = load_and_resize(path)
    for name, img in images.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=5)
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_filled = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
        black_img = np.zeros_like(opening)
        for cnt in contours_filled:
            cv2.drawContours(black_img, [cnt], 0, 255, -1)
            
        
        img_contours = np.zeros_like(binary)
        cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)

        show_changes("Seuillage + Morphologie", black_img, img_contours)

        result = img.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # Supprimer les petits objets
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),1)
        show_changes("Segmentation basique", img, result)
        output_path = os.path.join("results/seg_basic/", name)
        cv2.imwrite(output_path, result)

if __name__ == "__main__":
    path = "results/preprocessing/"
    segmentation_basic(path)