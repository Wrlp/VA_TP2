
import cv2
import numpy as np
import glob 
import os
import matplotlib.pyplot as plt

def load_and_resize(path, size=(600, 600)):
    images = {}
    for file in glob.glob(os.path.join(path, "*.png")):
        img = cv2.imread(file)
        img = cv2.resize(img, size)
        name = os.path.basename(file)  
        images[name] = img

    return images

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)

    return enhanced, blurred

def show_histogram(img, title="hist"):
    colors = ('b','g','r')
    plt.figure()
    plt.title(title)
    plt.xlabel("Intensité")
    plt.ylabel("Nombre de pixels")
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        print(f"Canal {col} - max: {np.max(hist):.3f}")
        plt.plot(hist, color=col)
    plt.xlim([0,256])
    plt.show()

def traitement(path):
    images = load_and_resize(path)
    for name, img in images.items():
        enhanced, blurred = preprocess(img)
        print(f"\nHistogramme pour {name}")
        show_histogram(img,"avant preprocess")
        show_histogram(enhanced,"après clahe")
        # show_image("Original", img)
        # show_image("CLAHE", enhanced)
        # show_image("Blurred", blurred)
        combined = np.hstack((img, enhanced))
        show_image("original vs clahe", combined)
        cv2.imwrite(f"results/{name}", combined)



