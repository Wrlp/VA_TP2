import cv2
import numpy as np
import os
import glob


def watershed_segmentation(input_path):
    """
    Applique Watershed sur :
    - une image
    - ou un dossier contenant des images
    """

    if os.path.isdir(input_path):

        images = glob.glob(os.path.join(input_path, "*.png"))

        if len(images) == 0:
            print("Aucune image trouvée dans le dossier.")
            return

        for img_path in images:
            print(f"Traitement : {img_path}")
            process_single_image(img_path)

    elif os.path.isfile(input_path):
        process_single_image(input_path)

    else:
        print("Chemin invalide.")


def process_single_image(image_path):

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"Impossible de lire {image_path}")
        return

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(distance, 0.4 * distance.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_bgr, markers)

    result_overlay = image_bgr.copy()
    result_overlay[markers == -1] = [0, 0, 255]

    num_grains = len(np.unique(markers)) - 2
    print(f"Nombre de grains détectés : {num_grains}")

    cv2.imshow("Watershed Segmentation", result_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_path = "results/preprocessing/"

    if not os.path.exists(test_path):
        os.makedirs(test_path)
        print(f"Dossier créé : {test_path}")
        
    print("Dossier courant :", os.getcwd())
    print("Existe ?", os.path.exists(test_path))

    print("Mode test autonome activé")
    watershed_segmentation(test_path)
